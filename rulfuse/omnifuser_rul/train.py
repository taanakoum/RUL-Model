"""One OmniFuser-RUL fit and one arm (docs/OMNIFUSER_RUL_PLAN.md §7).

A fit (fold, seed): the units' step tables standardized on the 8 training units; 30 epochs
of point-weighted MSE on y/H (Adam, learning rate 1e-4, batch 32, gradients clipped at 1);
after each epoch the weighted whole-life MAE (s) on the validation unit; the epoch with the
best mean of 3 consecutive values is kept; its predictions for every prediction point of
the validation and test units are written. Test predictions are written, never scored here
(D15: ``aggregate`` seals them until the freeze).

An arm is the set of fits of one input configuration over all folds and seeds; finished
fits are skipped, so an interrupted arm resumes.
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
    Layout,
    Standardizer,
    UnitSteps,
    Windows,
    collate,
)
from rulfuse.omnifuser_rul.model import OmniFuserRUL

FORCE = "force"  # the signal player; the other players are the store's image types


@dataclass(frozen=True)
class ArmSpec:
    """One input configuration: players (``force`` and/or image types), usage, dropout."""

    players: tuple[str, ...]
    usage: bool = True
    modality_dropout: float = 0.0
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    seeds: tuple[int, ...] = (0, 1, 2)
    input_steps: int = INPUT_STEPS

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(p for p in self.players if p != FORCE)

    @property
    def name(self) -> str:
        name = "+".join(self.players) if self.players else "usage-only"
        if not self.usage:
            name += "-nousage"
        if self.modality_dropout:
            name += f"-dropout{self.modality_dropout:g}"
        return name

    def signature(self) -> dict:
        """The settings that make a fit reusable, in JSON form (lists, not tuples)."""
        settings = asdict(self)
        settings.pop("seeds")
        return json.loads(json.dumps(settings))


@dataclass
class FitResult:
    fold: int
    seed: int
    selected_epoch: int
    validation_mae_s: list[float] = field(default_factory=list)
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


def _predict(
    model, windows: Windows, batch_size: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """(segment rows, predicted y/H) of every sample, in order."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    rows, out = [], []
    with torch.no_grad():
        for batch in DataLoader(windows, batch_size=batch_size, shuffle=False, collate_fn=collate):
            prediction = model(
                batch["force"].to(device),
                batch["images"].to(device),
                batch["image_indices"].to(device),
            )
            out.append(prediction.cpu().numpy())
            rows.append(batch["row"].numpy())
    return np.concatenate(rows), np.concatenate(out)


def weighted_mae_s(windows: Windows, prediction: np.ndarray, horizon_s: float) -> float:
    """Σ w |ŷ − y| over the samples (weights sum to 1 per unit; one unit here), in seconds."""
    y = np.array([windows.units[u].y_s[p] for u, p in windows.index])
    w = np.array([windows.units[u].weight[p] for u, p in windows.index])
    predicted = np.clip(prediction, 0.0, 1.0) * horizon_s
    return float(np.sum(w * np.abs(predicted - y)) / max(w.sum(), 1e-12))


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
    from torch.utils.data import DataLoader

    device = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    kinds_used = [all_kinds.index(k) for k in spec.kinds]
    train_units = [steps[u] for u in fold.train]
    standardizer = Standardizer.fit(train_units)
    layout = Layout(train_units[0].descriptors.shape[1])

    def windows(units: list[UnitSteps]) -> Windows:
        return Windows(units, standardizer, image_features, kinds_used, spec.input_steps)

    train, validation, test = (
        windows(train_units),
        windows([steps[fold.validation]]),
        windows([steps[fold.test]]),
    )
    model = OmniFuserRUL(
        layout,
        views=len(kinds_used),
        view_width=train.view_width,
        use_force=FORCE in spec.players,
        use_photos=bool(kinds_used),
        use_usage=spec.usage,
        modality_dropout=spec.modality_dropout,
        input_steps=spec.input_steps,
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=spec.learning_rate
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train, batch_size=spec.batch_size, shuffle=True, collate_fn=collate, generator=generator
    )
    history: list[float] = []
    states: list[dict] = []
    for epoch in range(1, spec.epochs + 1):
        model.train()
        for batch in loader:
            prediction = model(
                batch["force"].to(device),
                batch["images"].to(device),
                batch["image_indices"].to(device),
            )
            target = batch["y"].to(device) / horizon_s
            weight = batch["weight"].to(device)
            loss = (weight * (prediction - target) ** 2).sum() / weight.sum().clamp_min(1e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        _, validation_prediction = _predict(model, validation, spec.batch_size, device)
        history.append(weighted_mae_s(validation, validation_prediction, horizon_s))
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        log(
            f"  fold {fold_index} seed {seed} epoch {epoch}/{spec.epochs}: "
            f"validation MAE {history[-1]:.1f} s"
        )
    chosen = select_epoch(history)
    model.load_state_dict(states[chosen - 1])
    rows_out, units_out, roles, values = [], [], [], []
    for role, data in (("validation", validation), ("test", test)):
        rows, prediction = _predict(model, data, spec.batch_size, device)
        rows_out.append(rows)
        units_out += [data.units[0].name] * len(rows)
        roles += [role] * len(rows)
        values.append(np.clip(prediction, 0.0, 1.0) * horizon_s)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row": np.concatenate(rows_out),
            "unit": units_out,
            "role": roles,
            "fold": np.full(len(roles), fold_index),
            "seed": np.full(len(roles), seed),
            "prediction_s": np.concatenate(values),
        }
    )
    pq.write_table(table, out_dir / "predictions.parquet.tmp")
    (out_dir / "predictions.parquet.tmp").replace(out_dir / "predictions.parquet")
    result = FitResult(fold_index, seed, chosen, history, time.perf_counter() - started)
    record = {**asdict(result), "spec": spec.signature(), "fold_units": asdict(fold)}
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
