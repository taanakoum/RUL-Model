"""One RUL-Fuse fit: fold *i*, seed *s* (PROJECT_PLAN.md §7.2-§7.4; part P14).

Lifecycle (``fit``), writing into ``out_dir``:

1. **Setup.** Assert the roles are disjoint; seed NumPy and PyTorch from (fold, seed); write
   ``config.json``.
2. **Tokens.** Build them with training-only statistics and the cross-fitted readouts
   (:mod:`rulfuse.tokens`); write ``transforms.json``.
3. **Phase A** (when the dataset has an HI and ``phase_a`` is on): HI loss only, for
   ``phase_a_steps`` steps, with a linear warm-up.
4. **Phase B**: the full loss of §6.7, with the NLL ramped in and a cosine learning rate.
   The EMA weights (decay 0.99) restart at its first step. Every ``eval_every`` steps the
   EMA model is scored on the validation unit: the whole-life MAE in seconds with the loss's
   point weights (§4.3). Phase B stops after ``patience`` evaluations without improvement of
   the smoothed score (the mean of 3 consecutive evaluations).
5. **Restore** the EMA weights of the middle evaluation of the best 3; write ``model.pt``
   and ``selection.json``.
6. **Predict** every point of the validation, test and training units; write
   ``predictions.parquet``. Test rows are written but never read before the freeze (D15,
   :mod:`rulfuse.study`).
7. **Diagnostics** (§7.8, :mod:`rulfuse.diagnostics`); write ``diagnostics.json``, then
   ``DONE``.

A checkpoint is written at every evaluation. A rerun skips a fit with ``DONE`` and resumes
an interrupted one from its last checkpoint, with the same result as an uninterrupted fit.

One optimizer step (§7.3):

* the batch is ``copies`` augmented sequences of every training unit;
* the gradient norm is clipped at 1.0;
* AdamW;
* the EMA update.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold
from rulfuse.core.store import SegmentStore, stable_hash
from rulfuse.models.fusion import RULFuse, build_model
from rulfuse.models.heads import student_t_nll
from rulfuse.models.losses import monotonic_hinge, slope_consistency
from rulfuse.sequences import FoldTensors, Sequence, augmented, collate, make_sequence
from rulfuse.tokens import FoldData, build_fold_data

MAX_PAIR_OFFSET = 20


def seed_everything(fold_index: int, seed: int) -> np.random.Generator:
    value = 1000 * (fold_index + 1) + seed
    torch.manual_seed(value)
    # same seed -> same weights (P14): deterministic kernels (the gather's backward
    # accumulates in a fixed order); warn only where an op has no deterministic version
    torch.use_deterministic_algorithms(True, warn_only=True)
    return np.random.default_rng(value)


def fit_key(config: FitConfig) -> str:
    return stable_hash(config.to_dict())


# ----------------------------------------------------------------------------- loss


def _weighted(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (values * weight).sum() / weight.sum().clamp_min(1e-12)


def pair_terms(
    mu: torch.Tensor, batch: dict, horizon: float, config: FitConfig, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monotonic hinge and slope consistency (D22) on consecutive points and on points
    ``offset`` apart, both at most ``pair_max_fraction``·H apart; later point detached."""
    point, tau, y = batch["point"], batch["tau"] / horizon, batch["y"] / horizon
    earlier, later, elapsed, ok, below = [], [], [], [], []
    for k in sorted({1, offset}):
        if mu.shape[1] <= k:
            continue
        dt = tau[:, k:] - tau[:, :-k]
        valid = point[:, :-k] & point[:, k:] & (dt <= config.pair_max_fraction + 1e-9)
        earlier.append(mu[:, :-k].reshape(-1))
        later.append(mu[:, k:].reshape(-1))
        elapsed.append(dt.reshape(-1))
        ok.append(valid.reshape(-1))
        below.append((valid & (y[:, :-k] < 1.0 - 1e-6)).reshape(-1))
    if not earlier:
        zero = mu.sum() * 0
        return zero, zero
    e, lt, d = torch.cat(earlier), torch.cat(later), torch.cat(elapsed)
    hinge = monotonic_hinge(e, lt, torch.cat(ok))
    slope = slope_consistency(e, lt, d, torch.cat(below))
    return hinge, slope


def loss_terms(
    out: dict, batch: dict, horizon: float, config: FitConfig, phase: str, nll_w: float, offset: int
) -> tuple[torch.Tensor, dict[str, float]]:
    hi_valid = batch["hi_valid"]
    hi = (
        F.huber_loss(out["hi"][hi_valid], batch["hi"][hi_valid])
        if hi_valid.any()
        else out["hi"].sum() * 0
    )
    if phase == "A":
        return hi, {"hi": hi.item()}
    y = batch["y"] / horizon
    w = batch["weight"] * batch["point"]
    mu, sigma, nu = out["mu"], out["sigma"], out["nu"]
    mse = _weighted((mu - y) ** 2, w)
    nll = _weighted(student_t_nll(y, mu.detach(), sigma, nu), w)
    hinge, slope = pair_terms(mu, batch, horizon, config, offset)
    loss = (
        mse
        + nll_w * nll
        + config.mono_weight * hinge
        + config.cons_weight * slope
        + config.hi_weight * hi
    )
    parts = {
        k: v.item()
        for k, v in {"mse": mse, "nll": nll, "hinge": hinge, "slope": slope, "hi": hi}.items()
    }
    if "experts" in out:  # F0 deep supervision
        ex = out["experts"]
        terms = []
        for m in range(ex["mu"].shape[-1]):
            wm = w * ex["present"][..., m]
            if wm.sum() <= 0:
                continue
            mm, sm, nm = ex["mu"][..., m], ex["sigma"][..., m], ex["nu"][..., m]
            terms.append(
                _weighted((mm - y) ** 2, wm)
                + nll_w * _weighted(student_t_nll(y, mm.detach(), sm, nm), wm)
            )
        if terms:
            expert = torch.stack(terms).mean()
            loss = loss + config.expert_weight * expert
            parts["experts"] = expert.item()
    return loss, parts


# ----------------------------------------------------------------------------- predict


@dataclass
class Prediction:
    sequence: Sequence
    mu_s: np.ndarray
    sigma_s: np.ndarray
    nu: np.ndarray
    hi: np.ndarray
    alpha: np.ndarray | None = None


@torch.no_grad()
def predict(
    model: RULFuse, sequence: Sequence, tensors: FoldTensors, config: FitConfig, n_channels: int
) -> Prediction:
    model.eval()
    batch = collate([sequence], tensors, config, rng=None, n_channels=n_channels)
    out = model(batch, tensors)
    n = len(sequence.seg)
    h = model.horizon
    alpha = out["alpha"][0, :n].cpu().numpy() if "alpha" in out else None
    return Prediction(
        sequence,
        out["mu"][0, :n].cpu().numpy() * h,
        out["sigma"][0, :n].cpu().numpy() * h,
        out["nu"][0, :n].cpu().numpy(),
        out["hi"][0, :n].cpu().numpy(),
        alpha,
    )


def weighted_mae(pred: Prediction) -> float:
    s = pred.sequence
    w = s.weight * s.point
    if w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * np.abs(pred.mu_s - np.nan_to_num(s.y))) / w.sum())


def window_mae(pred: Prediction) -> float:
    s = pred.sequence
    keep = s.window
    return float(np.mean(np.abs(pred.mu_s[keep] - s.rul[keep]))) if keep.any() else float("nan")


# ----------------------------------------------------------------------------- fit


@dataclass
class FitState:
    step: int = 0
    evals: list[dict] = field(default_factory=list)
    curves: list[dict] = field(default_factory=list)
    best_smoothed: float = math.inf
    best_eval: int = -1
    since_best: int = 0


def _lr(step: int, config: FitConfig, phase_a: int, total: int) -> float:
    base = config.lr
    if step <= phase_a:
        return base * min(1.0, step / max(1, config.warmup_steps))
    b_step = step - phase_a
    b_total = max(1, total - phase_a)
    warm = 1.0 if phase_a else min(1.0, b_step / max(1, config.warmup_steps))
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, (b_step - 1) / b_total)))
    return warm * (config.min_lr + (base - config.min_lr) * cos)


def _ema_update(ema: RULFuse, model: RULFuse, decay: float) -> None:
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
            pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def _cpu_state(model: RULFuse) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fold_index(fold: Fold, units: list[str]) -> int:
    return units.index(fold.test)


def fit(
    store: SegmentStore,
    signal_features,
    image_features,
    fold: Fold,
    seed: int,
    config: FitConfig,
    out_dir: str | Path,
    device: str = "cpu",
    log=print,
    train_units: list[str] | None = None,
    data: FoldData | None = None,
    stop_after_steps: int | None = None,
) -> dict:
    """Run (or resume) one fit; returns its summary (``selection.json``)."""
    out = Path(out_dir)
    if (out / "DONE").is_file():
        return json.loads((out / "selection.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    roles = {fold.test, fold.validation}
    train = list(train_units or fold.train)
    if roles & set(train) or fold.test == fold.validation:
        raise ValueError("test, validation and training units must be disjoint")
    config.save(out / "config.json")
    rng = seed_everything(fold_index(fold, store.unit_names), seed)
    t0 = time.perf_counter()
    if data is None:
        data = build_fold_data(store, signal_features, image_features, fold, config)
    (out / "transforms.json").write_text(
        json.dumps({"hi_scale": data.hi_scale, **data.transforms}), encoding="utf-8"
    )
    horizon = store.settings.horizon_s
    n_ch = len(data.channels)
    model = build_model(config, data, horizon).to(device)
    ema = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_tensors = FoldTensors(data, train, device)
    unit_tensors = {u: FoldTensors(data, [u], device) for u in data.units}
    val_seq = make_sequence(data.units[fold.validation], None, config.w_far)
    phase_a = config.phase_a_steps if (config.phase_a and data.has_hi) else 0
    total = phase_a + (config.max_steps - config.phase_a_steps)
    state = FitState()
    snapshots: dict[int, dict] = {}
    ckpt = out / "checkpoint.pt"
    if ckpt.is_file():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        opt.load_state_dict(saved["opt"])
        state = saved["state"]
        snapshots = saved["snapshots"]
        rng.bit_generator.state = saved["np_rng"]
        torch.set_rng_state(saved["torch_rng"])
        log(f"[fit] resumed at step {state.step}")
    model.train()
    stopped = False
    while state.step < total and not stopped:
        state.step += 1
        step = state.step
        phase = "A" if step <= phase_a else "B"
        for g in opt.param_groups:
            g["lr"] = _lr(step, config, phase_a, total)
        if phase == "B" and step == phase_a + 1:
            ema.load_state_dict(model.state_dict())
        sequences = [
            augmented(data.units[u], rng, config) for u in train for _ in range(config.copies)
        ]
        batch = collate(sequences, train_tensors, config, rng=rng, n_channels=n_ch)
        offset = int(rng.integers(2, MAX_PAIR_OFFSET + 1))
        b_step = step - phase_a
        nll_w = config.nll_weight * min(1.0, max(0.0, (b_step - 1) / max(1, config.nll_ramp_steps)))
        model.train()
        result = model(batch, train_tensors)
        loss, parts = loss_terms(result, batch, horizon, config, phase, nll_w, offset)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
        opt.step()
        if phase == "B":
            _ema_update(ema, model, config.ema_decay)
        if step % 10 == 0 or step == 1:
            row = {"step": step, "phase": phase, "loss": loss.item(), "grad": grad, **parts}
            row["lr"] = opt.param_groups[0]["lr"]
            if "alpha" in result:
                a = result["alpha"][batch["point"]].mean(0).detach().cpu().numpy()
                row["alpha"] = [float(v) for v in a]
            state.curves.append(row)
        if phase == "B" and b_step % config.eval_every == 0:
            pred = predict(ema, val_seq, unit_tensors[fold.validation], config, n_ch)
            train_pred = [
                predict(
                    ema,
                    make_sequence(data.units[u], None, config.w_far),
                    unit_tensors[u],
                    config,
                    n_ch,
                )
                for u in train
            ]
            score = weighted_mae(pred)
            k = len(state.evals)
            state.evals.append(
                {
                    "step": step,
                    "val_mae_s": score,
                    "val_window_mae_s": window_mae(pred),
                    "train_mae_s": float(np.mean([weighted_mae(p) for p in train_pred])),
                    "coverage90": _coverage(pred, 0.9),
                }
            )
            snapshots[k] = _cpu_state(ema)
            if k >= 2:
                smoothed = float(np.mean([e["val_mae_s"] for e in state.evals[k - 2 : k + 1]]))
                if smoothed < state.best_smoothed - 1e-9:
                    state.best_smoothed, state.best_eval, state.since_best = smoothed, k - 1, 0
                else:
                    state.since_best += 1
                keep = {state.best_eval, k, k - 1}
                snapshots = {i: s for i, s in snapshots.items() if i in keep}
                stopped = state.since_best >= config.patience
            model.train()
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "state": state,
                    "snapshots": snapshots,
                    "np_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                },
                ckpt,
            )
        if stop_after_steps is not None and step >= stop_after_steps:
            return {"interrupted_at": step}
    if state.best_eval < 0:  # fewer than 3 evaluations
        scores = [e["val_mae_s"] for e in state.evals]
        state.best_eval = int(np.argmin(scores)) if scores else -1
    if state.best_eval >= 0 and state.best_eval in snapshots:
        ema.load_state_dict(snapshots[state.best_eval])
    selected = ema
    torch.save(selected.state_dict(), out / "model.pt")
    elapsed = time.perf_counter() - t0
    preds = {}
    for u in [fold.validation, fold.test, *train]:
        if u in data.units:
            seq = make_sequence(data.units[u], None, config.w_far)
            preds[u] = predict(selected, seq, unit_tensors[u], config, n_ch)
    roles_map = {fold.validation: "validation", fold.test: "test"}
    write_predictions(out / "predictions.parquet", preds, roles_map, data)
    from rulfuse.diagnostics import fit_diagnostics

    diag = fit_diagnostics(selected, data, fold, train, config, state, preds, unit_tensors, store)
    (out / "diagnostics.json").write_text(
        json.dumps(diag, indent=1, default=float), encoding="utf-8"
    )
    selection = {
        "fold_test": fold.test,
        "fold_validation": fold.validation,
        "seed": seed,
        "config_key": fit_key(config),
        "selected_eval": state.best_eval,
        "selected_step": state.evals[state.best_eval]["step"] if state.best_eval >= 0 else None,
        "evaluations": state.evals,
        "steps_run": state.step,
        "seconds": elapsed,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "device": device,
    }
    (out / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    (out / "curves.json").write_text(json.dumps(state.curves), encoding="utf-8")
    ckpt.unlink(missing_ok=True)
    (out / "DONE").write_text("ok", encoding="utf-8")
    kept = state.evals[state.best_eval]["val_mae_s"] if state.best_eval >= 0 else float("nan")
    log(
        f"[fit] {fold.test} seed {seed}: step {selection['selected_step']} kept, "
        f"validation MAE {kept:.1f} s, {elapsed:.0f} s"
    )
    return selection


def _coverage(pred: Prediction, level: float) -> float:
    from scipy.stats import t as student

    s = pred.sequence
    keep = s.window
    if not keep.any():
        return float("nan")
    q = student.ppf(0.5 + level / 2, pred.nu[keep])
    lo, hi = pred.mu_s[keep] - q * pred.sigma_s[keep], pred.mu_s[keep] + q * pred.sigma_s[keep]
    return float(np.mean((s.rul[keep] >= lo) & (s.rul[keep] <= hi)))


def write_predictions(
    path: Path, preds: dict[str, Prediction], roles: dict, data: FoldData
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    mean, sd = data.hi_scale
    cols: dict[str, list] = {
        k: []
        for k in (
            "unit",
            "role",
            "segment",
            "tau_s",
            "elapsed_s",
            "window",
            "rul_s",
            "y_s",
            "mu_s",
            "sigma_s",
            "nu",
            "hi_hat",
        )
    }
    alpha_cols: list[list[float]] = []
    for unit, p in preds.items():
        s = p.sequence
        keep = s.point
        n = int(keep.sum())
        cols["unit"] += [unit] * n
        cols["role"] += [roles.get(unit, "train")] * n
        cols["segment"] += s.seg[keep].tolist()
        cols["tau_s"] += s.tau[keep].tolist()
        cols["elapsed_s"] += s.elapsed[keep].tolist()
        cols["window"] += s.window[keep].tolist()
        cols["rul_s"] += s.rul[keep].tolist()
        cols["y_s"] += s.y[keep].tolist()
        cols["mu_s"] += p.mu_s[keep].tolist()
        cols["sigma_s"] += p.sigma_s[keep].tolist()
        cols["nu"] += p.nu[keep].tolist()
        cols["hi_hat"] += (p.hi[keep] * sd + mean).tolist()
        if p.alpha is not None:
            alpha_cols += p.alpha[keep].tolist()
    table = {k: v for k, v in cols.items()}
    if alpha_cols:
        a = np.asarray(alpha_cols)
        for i, name in enumerate(("alpha_usage", "alpha_signal", "alpha_image")):
            table[name] = a[:, i].tolist()
    pq.write_table(pa.table(table), path)
