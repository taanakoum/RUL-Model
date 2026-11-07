"""One OmniFuser-RUL fit and one arm (docs/OMNIFUSER_RUL_PLAN.md §7; version 3).

A fit (fold, seed):

1. Step tables of all units, standardized on the 8 training units; per unit a table of
   photo composites; everything on the device (``FitTensors``).
2. Each epoch draws ``pairs_per_epoch`` pairs (earlier, later) of prediction points of the
   training units (``data.sample_pairs``: lag 1, the previous segment, half of the time;
   up to ``max_lag`` steps otherwise). Per batch of ``batch_size`` windows (half as many
   pairs):

   * the memory runs over the training units' whole histories (O8);
   * OmniFuser encodes the window of each point of each pair;
   * the loss (``pair_loss``) is the point-weighted MSE on y/H of both points
     + ``mono_weight`` × the monotonic hinge relu(ŷ_later − ŷ_earlier)
     + ``slope_weight`` × the slope consistency ((ŷ_earlier − ŷ_later) − Δτ/H)² where the
     earlier target is below the cap (O9, D22). In both temporal terms the later
     prediction is the fixed reference (detached): they move the earlier prediction only,
     so an earlier prediction the model cannot correct never drags later ones with it.

   Adam, learning rate 1e-4, gradients clipped at 1 (OmniFuser's settings).
3. After each epoch, the weighted whole-life MAE (s) on the validation unit's points of the
   evaluation grid (the project's 10 s points). The epoch at the centre of the 3 consecutive
   values with the lowest mean is kept.
4. Its predictions for every prediction point of the validation and test units are written,
   with ``on_grid`` marking the evaluation points. Test predictions are written, never
   scored here (D15: ``aggregate`` seals them until the freeze).

An arm is the set of fits of one input configuration over all folds and seeds; finished
fits with the same settings are skipped, so an interrupted arm resumes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rulfuse.core.folds import Fold, leave_one_unit_out
from rulfuse.core.store import SegmentStore, stable_hash
from rulfuse.omnifuser_rul.data import (
    INPUT_STEPS,
    FitTensors,
    Layout,
    Standardizer,
    UnitInputs,
    UnitSteps,
    sample_pairs,
    unit_inputs,
)
from rulfuse.omnifuser_rul.model import DROP_FORCE, DROP_PHOTO, NO_DROP, MemorySpec, OmniFuserRUL

FORCE = "force"  # the signal player; the other players are the store's image types


@dataclass(frozen=True)
class ArmSpec:
    """One input configuration and its training settings."""

    players: tuple[str, ...]
    usage: bool = True
    modality_dropout: float = 0.0
    epochs: int = 30
    batch_size: int = 32  # windows per step (batch_size / 2 pairs)
    learning_rate: float = 1e-4
    seeds: tuple[int, ...] = (0, 1, 2)
    input_steps: int = INPUT_STEPS
    memory: bool = True  # O8
    memory_hidden: int = 64
    memory_block: int = 10
    mono_weight: float = 1.0 / 15.0  # O9: relu(ŷ_later − ŷ_earlier); pulls ≤ about H/15
    slope_weight: float = 0.05  # O9: slope consistency below the cap
    max_lag: int = 200  # steps
    pairs_per_epoch: int = 2048

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(p for p in self.players if p != FORCE)

    @property
    def temporal(self) -> bool:
        return bool(self.mono_weight or self.slope_weight)

    @property
    def name(self) -> str:
        name = "+".join(self.players) if self.players else "usage-only"
        if not self.usage:
            name += "-nousage"
        if self.modality_dropout:
            name += f"-dropout{self.modality_dropout:g}"
        if not self.memory:
            name += "-nomemory"
        if not self.temporal:
            name += "-notemporal"
        return name

    def signature(self) -> dict:
        """The settings that make a fit reusable, in JSON form (lists, not tuples)."""
        settings = asdict(self)
        settings.pop("seeds")
        return json.loads(json.dumps(settings))

    def memory_spec(self) -> MemorySpec | None:
        return MemorySpec(self.memory_hidden, self.memory_block) if self.memory else None


@dataclass
class FitResult:
    fold: int
    seed: int
    selected_epoch: int
    validation_mae_s: list[float] = field(default_factory=list)
    train_loss: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def _device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def select_epoch(validation_mae: Sequence[float]) -> int:
    """1-based epoch at the centre of the 3 consecutive values with the lowest mean
    (PROJECT_PLAN.md §4.3); with fewer than 3 values, the lowest one."""
    values = np.asarray(validation_mae, dtype=float)
    if len(values) < 3:
        return int(np.argmin(values)) + 1
    triples = (values[:-2] + values[1:-1] + values[2:]) / 3.0
    return int(np.argmin(triples)) + 2


def predict_unit(
    model: OmniFuserRUL,
    tensors: FitTensors,
    index: int,
    steps: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """ŷ/H at the given steps of unit ``index`` of ``tensors`` (evaluation mode)."""
    import torch

    model.eval()
    device = tensors.force.device
    out = []
    with torch.no_grad():
        memory = None
        if model.memory_spec is not None:
            force, composites, rows, lengths = tensors.memory_inputs()
            memory = model.memory_states(
                force[index : index + 1],
                composites[index : index + 1],
                rows[index : index + 1],
                lengths[index : index + 1],
            )[0]
        for start in range(0, len(steps), batch_size):
            part = torch.as_tensor(steps[start : start + batch_size], device=device)
            unit = torch.full_like(part, index)
            force_w, images, image_indices = tensors.windows(unit, part)
            state = None if memory is None else memory[part]
            out.append(model(force_w, images, image_indices, state).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def weighted_mae_s(
    unit: UnitSteps, steps: np.ndarray, prediction: np.ndarray, horizon_s: float
) -> float:
    """Σ w |ŷ − y| / Σ w over the given steps, in seconds: the unit's grid weights, or its
    point weights when it has no grid point."""
    w = unit.grid_weight[steps] if unit.grid.any() else unit.weight[steps]
    predicted = np.clip(prediction, 0.0, 1.0) * horizon_s
    return float(np.sum(w * np.abs(predicted - unit.y_s[steps])) / max(w.sum(), 1e-12))


def pair_loss(
    prediction,
    target,
    weight,
    elapsed,
    paired,
    mono_weight: float,
    slope_weight: float,
):
    """The training loss of a batch of k pairs (O9, D22). ``prediction``, ``target`` and
    ``weight`` hold the k earlier points then the k later points (y/H); ``elapsed`` is each
    pair's Δτ/H and ``paired`` marks the pairs whose later step is after the earlier one.
    Returns (loss, parts). The temporal terms reach the earlier predictions only
    (``rulfuse.models.losses``): the later half's gradient is the MSE's alone."""
    from rulfuse.models.losses import monotonic_hinge, slope_consistency

    mse = (weight * (prediction - target) ** 2).sum() / weight.sum().clamp_min(1e-12)
    k = len(prediction) // 2
    p_earlier, p_later = prediction[:k], prediction[k:]
    hinge = monotonic_hinge(p_earlier, p_later, paired)
    below_cap = paired & (target[:k] < 1.0 - 1e-9)
    slope = slope_consistency(p_earlier, p_later, elapsed, below_cap)
    loss = mse + mono_weight * hinge + slope_weight * slope
    return loss, {"mse": mse, "hinge": hinge, "slope": slope}


def _memory_for_batch(model, tensors: FitTensors, drop):
    """Memory states of the training units, (variants, U, n, hidden): the full inputs, and
    without force or photos when modality dropout uses them."""
    if model.memory_spec is None:
        return None
    force, composites, rows, lengths = tensors.memory_inputs()
    variants = [NO_DROP]
    if drop is not None:
        variants += [DROP_FORCE, DROP_PHOTO]
    import torch

    return torch.stack([model.memory_states(force, composites, rows, lengths, v) for v in variants])


def fit(
    steps: dict[str, UnitSteps],
    image_features: np.ndarray,
    all_kinds: Sequence[str],
    spec: ArmSpec,
    fold: Fold,
    fold_index: int,
    seed: int,
    horizon_s: float,
    out_dir: Path,
    device: str = "auto",
    log=print,
) -> FitResult:
    import torch

    device = _device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, fold_index])
    started = time.perf_counter()
    kinds_used = [all_kinds.index(k) for k in spec.kinds]
    train_steps = [steps[u] for u in fold.train]
    standardizer = Standardizer.fit(train_steps)
    layout = Layout(train_steps[0].descriptors.shape[1])

    def inputs(unit_steps: list[UnitSteps]) -> list[UnitInputs]:
        return [unit_inputs(s, standardizer, image_features, kinds_used) for s in unit_steps]

    train_inputs = inputs(train_steps)
    val_inputs, test_inputs = inputs([steps[fold.validation]]), inputs([steps[fold.test]])
    train_t = FitTensors(train_inputs, spec.input_steps, device)
    val_t = FitTensors(val_inputs, spec.input_steps, device)
    test_t = FitTensors(test_inputs, spec.input_steps, device)
    view_width = image_features.shape[1]
    model = OmniFuserRUL(
        layout,
        views=len(kinds_used),
        view_width=view_width,
        use_force=FORCE in spec.players,
        use_photos=bool(kinds_used),
        use_usage=spec.usage,
        modality_dropout=spec.modality_dropout,
        input_steps=spec.input_steps,
        memory=spec.memory_spec(),
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=spec.learning_rate
    )
    # Per training step: targets, weights and times of every (unit, step), on the device.
    y = torch.zeros(len(train_inputs), int(train_t.lengths.max()), device=device)
    w, tau = torch.zeros_like(y), torch.zeros_like(y)
    for i, u in enumerate(train_inputs):
        y[i, : u.n] = torch.as_tensor(u.steps.y_s[: u.n] / horizon_s, device=device)
        w[i, : u.n] = torch.as_tensor(u.steps.weight[: u.n], device=device)
        tau[i, : u.n] = torch.as_tensor(u.steps.tau_s[: u.n] / horizon_s, device=device)
    pairs_per_batch = max(spec.batch_size // 2, 1)
    val_unit = val_inputs[0].steps
    val_steps = np.flatnonzero(val_unit.grid[: val_inputs[0].n])
    if not len(val_steps):  # no evaluation-grid point: select on every point
        val_steps = np.arange(val_inputs[0].n)
    history: list[float] = []
    losses: list[dict] = []
    states: list[dict] = []
    for epoch in range(1, spec.epochs + 1):
        model.train()
        pairs = sample_pairs(train_inputs, spec.pairs_per_epoch, spec.max_lag, rng)
        sums = {"mse": 0.0, "hinge": 0.0, "slope": 0.0, "batches": 0}
        for start in range(0, len(pairs), pairs_per_batch):
            part = torch.as_tensor(pairs[start : start + pairs_per_batch], device=device)
            unit, earlier, later = part[:, 0], part[:, 1], part[:, 2]
            both_unit = torch.cat([unit, unit])
            both_step = torch.cat([earlier, later])
            drop = model.draw_dropout(len(both_unit) // 2, device)
            drop_both = None if drop is None else torch.cat([drop, drop])
            memory = _memory_for_batch(model, train_t, drop)
            state = None
            if memory is not None:
                variant = torch.zeros_like(both_unit) if drop_both is None else drop_both
                state = memory[variant, both_unit, both_step]
            force_w, images, image_indices = train_t.windows(both_unit, both_step)
            prediction = model(force_w, images, image_indices, state, drop_both)
            loss, parts = pair_loss(
                prediction,
                y[both_unit, both_step],
                w[both_unit, both_step],
                tau[unit, later] - tau[unit, earlier],
                later > earlier,
                spec.mono_weight,
                spec.slope_weight,
            )
            mse, hinge, slope = parts["mse"], parts["hinge"], parts["slope"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["mse"] += float(mse.detach())
            sums["hinge"] += float(hinge.detach())
            sums["slope"] += float(slope.detach())
            sums["batches"] += 1
        validation_prediction = predict_unit(model, val_t, 0, val_steps, spec.batch_size)
        history.append(weighted_mae_s(val_unit, val_steps, validation_prediction, horizon_s))
        batches = max(sums.pop("batches"), 1)
        losses.append({k: round(v / batches, 6) for k, v in sums.items()})
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log(
            f"  fold {fold_index} seed {seed} epoch {epoch}/{spec.epochs}: validation MAE "
            f"{history[-1]:.1f} s; loss mse {losses[-1]['mse']:.4f} hinge "
            f"{losses[-1]['hinge']:.5f} slope {losses[-1]['slope']:.5f}"
        )
    chosen = select_epoch(history)
    model.load_state_dict(states[chosen - 1])
    columns: dict[str, list] = {k: [] for k in ("row", "unit", "role", "on_grid", "prediction_s")}
    for role, tensors, data in (
        ("validation", val_t, val_inputs[0]),
        ("test", test_t, test_inputs[0]),
    ):
        every = np.arange(data.n)
        prediction = predict_unit(model, tensors, 0, every, spec.batch_size)
        columns["row"] += data.steps.rows[every].tolist()
        columns["unit"] += [data.name] * len(every)
        columns["role"] += [role] * len(every)
        columns["on_grid"] += data.steps.grid[every].tolist()
        columns["prediction_s"] += (np.clip(prediction, 0.0, 1.0) * horizon_s).tolist()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(columns["row"])
    table = pa.table(
        {
            "row": np.asarray(columns["row"], dtype=np.int64),
            "unit": columns["unit"],
            "role": columns["role"],
            "fold": np.full(n, fold_index),
            "seed": np.full(n, seed),
            "on_grid": np.asarray(columns["on_grid"], dtype=bool),
            "prediction_s": np.asarray(columns["prediction_s"], dtype=np.float64),
        }
    )
    pq.write_table(table, out_dir / "predictions.parquet.tmp")
    (out_dir / "predictions.parquet.tmp").replace(out_dir / "predictions.parquet")
    result = FitResult(fold_index, seed, chosen, history, losses, time.perf_counter() - started)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record = {
        **asdict(result),
        "spec": spec.signature(),
        "fold_units": asdict(fold),
        "trainable_parameters": int(parameters),
        "train_points": int(sum(u.n for u in train_inputs)),
    }
    (out_dir / "fit.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return result


def arm_dir(root: Path, store: SegmentStore, spec: ArmSpec) -> Path:
    return root / f"{store.dataset}-{store.signature}" / spec.name


def run_arm(
    store: SegmentStore,
    steps: dict[str, UnitSteps],
    image_features: np.ndarray,
    all_kinds: Sequence[str],
    spec: ArmSpec,
    root: Path,
    folds: Sequence[int] | None = None,
    device: str = "auto",
    log=print,
) -> Path:
    """Every (fold, seed) fit of an arm not finished yet with the same settings."""
    unknown = set(spec.kinds) - set(all_kinds)
    if unknown:
        raise ValueError(f"image types {sorted(unknown)} are not in the store")
    all_folds = leave_one_unit_out(store.unit_names)
    folder = arm_dir(root, store, spec)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "arm.json").write_text(
        json.dumps(
            {"spec": asdict(spec), "store": store.signature, "key": stable_hash(spec.signature())},
            indent=1,
        ),
        encoding="utf-8",
    )
    for index, fold in enumerate(all_folds, start=1):
        if folds is not None and index not in folds:
            continue
        for seed in spec.seeds:
            out = folder / f"fold-{index:02d}" / f"seed-{seed}"
            done = out / "fit.json"
            if (
                done.is_file()
                and json.loads(done.read_text(encoding="utf-8"))["spec"] == spec.signature()
            ):
                log(f"[omnifuser-rul] {spec.name} fold {index} seed {seed}: done, skipped")
                continue
            log(f"[omnifuser-rul] {spec.name} fold {index} (test {fold.test}) seed {seed}")
            result = fit(
                steps,
                image_features,
                all_kinds,
                spec,
                fold,
                index,
                seed,
                store.settings.horizon_s,
                out,
                device,
                log,
            )
            log(f"[omnifuser-rul] epoch {result.selected_epoch} kept; {result.seconds:.0f} s")
    return folder
