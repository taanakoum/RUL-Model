"""Per-fit diagnostics: is the model learning or memorizing? (PROJECT_PLAN.md §7.8, D25)

Written to ``diagnostics.json`` by every fit; only the validation and training units are
read (D15).

1. **Learning curves:** the training loss terms and the validation MAE per evaluation.
   The memorization warning fires when the selected evaluation is among the first 3 while
   the training loss keeps falling.
2. **Against the usage rules:** validation window MAE against B2 and against the constant
   B0 = H/2 on the same unit.
3. **Usage sensitivity s_U:** 20 % of the validation unit's eligible recordings are deleted
   (never the baseline or failure recordings). The true RUL at the later points does not
   change, but the elapsed time drops by Δ. s_U = Σ(rise of the prediction) / ΣΔ: 0 means
   the model reads the sensors only, 1 means it counts time only.
4. **Modality reliance:** validation window MAE with each modality masked, at the selected
   weights.
5. **Calibration:** validation coverage of the raw 90 % intervals per evaluation.
6. **Head health:** shares of μ at 0 or H, of σ at its floor, and F0's mean weights.

**Added after baseline B6 (D25, from its training and validation curves only, not its test
scores).** B6's training error fell about 50 times while its validation MAE stayed flat,
and some tools stayed at the cap. So each fit also reports:

7. **generalization gap:** the training units' weighted MAE against the validation unit's,
   at the selected evaluation. A ratio above 3 warns;
8. **selection noise:** the sd of successive differences of the validation curve. With this
   much noise, a single minimum is optimistic;
9. **cap saturation:** the share of validation window points predicted within H/15 of the
   cap while the true RUL is below H/2.
"""

from __future__ import annotations

import numpy as np

from rulfuse.baselines import b2
from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold
from rulfuse.core.store import SegmentStore
from rulfuse.models.heads import SIGMA_FLOOR
from rulfuse.sequences import make_sequence
from rulfuse.tokens import FoldData

S_U_DELETE = 0.2
GAP_WARN_RATIO = 3.0


def deletion_plan(unit, fraction: float, seed: int = 0) -> list[int]:
    failure = unit.failure_rec
    eligible = [p for p in range(len(unit.rec_start)) if not unit.rec_baseline[p] and p < failure]
    rng = np.random.default_rng(seed)
    n = int(round(fraction * len(eligible)))
    gone = set(rng.choice(eligible, size=n, replace=False).tolist()) if n else set()
    return [p for p in range(len(unit.rec_start)) if p not in gone]


def usage_sensitivity(model, data: FoldData, unit_name: str, tensors, config, n_ch) -> float:
    from rulfuse.train import predict

    unit = data.units[unit_name]
    full = predict(model, make_sequence(unit, None, config.w_far), tensors, config, n_ch)
    plan = deletion_plan(unit, S_U_DELETE)
    cut = predict(model, make_sequence(unit, plan, config.w_far), tensors, config, n_ch)
    by_seg = {int(s): i for i, s in enumerate(full.sequence.seg)}
    rise, delta = 0.0, 0.0
    for j, s in enumerate(cut.sequence.seg):
        i = by_seg[int(s)]
        d = full.sequence.tau[i] - cut.sequence.tau[j]
        if d > 1e-6 and full.sequence.window[i]:
            rise += cut.mu_s[j] - full.mu_s[i]
            delta += d
    return float(rise / delta) if delta > 0 else float("nan")


def fit_diagnostics(
    model,
    data: FoldData,
    fold: Fold,
    train: list[str],
    config: FitConfig,
    state,
    preds: dict,
    tensors: dict,
    store: SegmentStore,
) -> dict:
    from rulfuse.train import predict, window_mae

    horizon = store.settings.horizon_s
    n_ch = len(data.channels)
    val = preds[fold.validation]
    seq = val.sequence
    evals = state.evals
    k = state.best_eval
    curve = [e["val_mae_s"] for e in evals]
    train_loss = [c["loss"] for c in state.curves if c["phase"] == "B"]
    falling = len(train_loss) > 2 and train_loss[-1] < 0.8 * train_loss[0]
    failures = np.array([store.units[u]["failure_s"] for u in train if store.units[u]["failure_s"]])
    keep = seq.window
    y = seq.rul[keep]
    b2_mae = float(np.mean(np.abs(b2(failures, seq.tau[keep], horizon) - y))) if keep.any() else None
    b0_mae = float(np.mean(np.abs(horizon / 2 - y))) if keep.any() else None
    model_mae = window_mae(val)
    reliance = {}
    for name, overrides in (("signal", {"players": tuple(p for p in config.players if p != "signal")}),
                            ("images", {"players": tuple(p for p in config.players if p == "signal")}),
                            ("usage", {"usage": False})):
        masked = config.with_overrides(overrides)
        old = model.config
        model.config = masked
        try:
            reliance[name] = window_mae(
                predict(model, make_sequence(data.units[fold.validation], None, config.w_far),
                        tensors[fold.validation], masked, n_ch)
            )
        finally:
            model.config = old
    gap = None
    if k >= 0:
        e = evals[k]
        gap = {
            "train_mae_s": e["train_mae_s"],
            "val_mae_s": e["val_mae_s"],
            "ratio": e["val_mae_s"] / max(e["train_mae_s"], 1e-9),
        }
    noise = float(np.std(np.diff(curve))) if len(curve) > 2 else None
    low = keep & (np.nan_to_num(seq.rul) < horizon / 2)
    saturation = float(np.mean(val.mu_s[low] >= horizon - horizon / 15)) if low.any() else None
    mu_n = val.mu_s[seq.point] / horizon
    sigma_n = val.sigma_s[seq.point] / horizon
    warnings = []
    if 0 <= k < 3 and falling:
        warnings.append("memorization: the selected evaluation is among the first 3")
    if b2_mae is not None and model_mae > b2_mae:
        warnings.append("validation MAE above B2")
    if b0_mae is not None and model_mae > b0_mae:
        warnings.append("validation MAE above the constant B0")
    if gap and gap["ratio"] > GAP_WARN_RATIO:
        warnings.append(f"generalization gap: validation / training MAE = {gap['ratio']:.1f}")
    if saturation is not None and saturation > 0.5:
        warnings.append("cap saturation on the validation unit")
    alpha = None
    if val.alpha is not None:
        alpha = dict(zip(("usage", "signal", "image"), val.alpha[seq.point].mean(0).tolist(), strict=True))
    return {
        "validation_unit": fold.validation,
        "validation_window_mae_s": model_mae,
        "b2_window_mae_s": b2_mae,
        "b0_window_mae_s": b0_mae,
        "usage_sensitivity": usage_sensitivity(
            model, data, fold.validation, tensors[fold.validation], config, n_ch
        ),
        "modality_masked_window_mae_s": reliance,
        "coverage90_by_eval": [e["coverage90"] for e in evals],
        "head": {
            "mu_at_bounds": float(np.mean((mu_n < 0.01) | (mu_n > 0.99))) if len(mu_n) else None,
            "sigma_at_floor": float(np.mean(sigma_n < SIGMA_FLOOR * 1.1)) if len(sigma_n) else None,
            "alpha": alpha,
        },
        "generalization_gap": gap,
        "selection_noise_s": noise,
        "cap_saturation": saturation,
        "warnings": warnings,
    }
