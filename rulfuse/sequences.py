"""Model sequences from units: trajectory augmentation, targets and batches (PROJECT_PLAN.md
§6.8, §7.3; parts P10, P13).

**Trajectory augmentation** (the main guard against the usage shortcut). Each recording
from the first non-baseline one up to, not including, the failure recording is:

* deleted with probability ρ,
* duplicated with probability ρ,

with ρ ~ U[0, ρ_max] per sequence. Times are recomputed:

* every later recording moves by the durations removed or added before it;
* unrecorded gaps keep their length;
* the failure time moves with its recording, so the unit wears faster or slower per
  recording and elapsed time stops predicting RUL;
* images move with the recording they follow (and are deleted or duplicated with it);
* every target is recomputed from the new times.

The baseline recordings and the failure recording are never touched. With cached tokens
all of this is index manipulation.

**Targets** of a sequence follow :mod:`rulfuse.core.targets`: a point is a segment end at
or before the failure, y = min(τ* − τ, H), and the window is RUL ≤ H.

**Point weights** (§7.3) are v = 1 in the window and w_far elsewhere, normalized so that
every sequence (unit) weighs the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rulfuse.config import FitConfig
from rulfuse.core.schema import TIME_TOLERANCE_S
from rulfuse.tokens import FoldData, UnitData


@dataclass
class Sequence:
    unit: str
    seg: np.ndarray  # (L,) rows of unit.signal
    tau: np.ndarray
    elapsed: np.ndarray
    entry: np.ndarray  # recording position in the (augmented) sequence
    present: np.ndarray
    point: np.ndarray
    rul: np.ndarray
    y: np.ndarray  # seconds, capped at H
    window: np.ndarray
    weight: np.ndarray
    hi: np.ndarray
    oracle: np.ndarray
    img: np.ndarray  # (I,) rows of unit.images
    img_time: np.ndarray
    img_entry: np.ndarray
    failure_s: float | None


def trajectory_plan(unit: UnitData, rng: np.random.Generator, rho_max: float) -> list[int]:
    """Recording positions in their new order (deleted ones left out, duplicates repeated)."""
    failure = unit.failure_rec
    rho = rng.uniform(0.0, rho_max) if rho_max > 0 else 0.0
    plan: list[int] = []
    for p in range(len(unit.rec_start)):
        if rho > 0 and not unit.rec_baseline[p] and p < failure:
            u = rng.random()
            if u < rho:
                continue
            plan.append(p)
            if u < 2 * rho:
                plan.append(p)
        else:
            plan.append(p)
    return plan


def make_sequence(
    unit: UnitData,
    plan: list[int] | None = None,
    w_far: float = 0.3,
) -> Sequence:
    """The sequence of ``unit`` with recordings in ``plan`` order (identity when None)."""
    order = list(range(len(unit.rec_start))) if plan is None else list(plan)
    duration = unit.rec_end - unit.rec_start
    starts = np.zeros(len(order))
    shift = 0.0
    seen: set[int] = set()
    previous = -1
    for k, p in enumerate(order):
        # recordings dropped between the previous kept one and p shorten everything after
        for gone in range(previous + 1, p):
            if gone not in seen:
                shift -= duration[gone]
        if p in seen:  # a duplicate starts where its copy ended; later recordings move on
            starts[k] = starts[k - 1] + duration[p]
            shift += duration[p]
        else:
            starts[k] = unit.rec_start[p] + shift
        seen.add(p)
        previous = max(previous, p)
    failure = None
    if unit.failure_s is not None:
        f = unit.failure_rec
        k = order.index(f) if f in order else None
        if k is None:
            raise ValueError("the failure recording must stay in the plan")
        failure = unit.failure_s - unit.rec_start[f] + starts[k]
    seg_parts, entry_parts, tau_parts = [], [], []
    by_rec = unit.rec_segments
    for k, p in enumerate(order):
        rows = by_rec[p]
        seg_parts.append(rows)
        entry_parts.append(np.full(len(rows), k))
        tau_parts.append(starts[k] + unit.seg_elapsed[rows])
    seg = np.concatenate(seg_parts) if seg_parts else np.zeros(0, np.int64)
    entry = np.concatenate(entry_parts) if entry_parts else np.zeros(0, np.int64)
    tau = np.concatenate(tau_parts) if tau_parts else np.zeros(0)
    horizon = unit.horizon_s
    if failure is None:
        point = np.zeros(len(seg), bool)
        rul = np.full(len(seg), np.nan)
    else:
        point = tau <= failure + TIME_TOLERANCE_S
        rul = np.where(point, np.maximum(failure - tau, 0.0), np.nan)
    y = np.where(point, np.minimum(rul, horizon), np.nan)
    window = point & (rul <= horizon + TIME_TOLERANCE_S)
    v = np.where(window, 1.0, w_far) * point
    weight = v / v.sum() if v.sum() > 0 else v
    # oracle input: the causal HI of the recording before this one
    hi_prev = np.zeros(len(order))
    for k in range(1, len(order)):
        value = unit.rec_hi_causal[order[k - 1]]
        hi_prev[k] = value if np.isfinite(value) else 0.0
    oracle = hi_prev[entry] if len(entry) else np.zeros(0)
    # images move with their recording
    ends = starts + duration[order] if len(order) else np.zeros(0)
    parts = [(k, unit.rec_images[p]) for k, p in enumerate(order) if len(unit.rec_images[p])]
    img = np.concatenate([rows for _, rows in parts]) if parts else np.zeros(0, np.int64)
    img_entry = (
        np.concatenate([np.full(len(rows), k) for k, rows in parts]) if parts else np.zeros(0)
    ).astype(np.int64)
    img_time_a = (ends[img_entry] + unit.img_offset[img]) if len(img) else np.zeros(0)
    img = img.astype(np.int64)
    keep = np.argsort(img_time_a, kind="stable")
    seq = Sequence(
        unit=unit.name,
        seg=seg.astype(np.int64),
        tau=tau,
        elapsed=unit.seg_elapsed[seg],
        entry=entry.astype(np.int64),
        present=unit.seg_present[seg],
        point=point,
        rul=rul,
        y=y,
        window=window,
        weight=weight,
        hi=unit.seg_hi[seg],
        oracle=oracle,
        img=img[keep],
        img_time=img_time_a[keep],
        img_entry=img_entry[keep],
        failure_s=failure,
    )
    return seq


def augmented(unit: UnitData, rng: np.random.Generator, config: FitConfig) -> Sequence:
    use = config.trajectory_aug and config.regularize
    plan = trajectory_plan(unit, rng, config.rho_max) if use else None
    return make_sequence(unit, plan, config.w_far)


# ----------------------------------------------------------------------------- tensors


class FoldTensors:
    """The base tokens of some units, concatenated, on one device."""

    def __init__(self, data: FoldData, names: list[str], device: str | torch.device) -> None:
        self.names = list(names)
        self.device = torch.device(device)
        units = [data.units[n] for n in self.names]
        self.signal_offset = dict(
            zip(self.names, np.cumsum([0] + [len(u.signal) for u in units])[:-1], strict=True)
        )
        self.image_offset = dict(
            zip(self.names, np.cumsum([0] + [len(u.images) for u in units])[:-1], strict=True)
        )
        self.signal = torch.from_numpy(np.concatenate([u.signal for u in units])).to(self.device)
        self.present_np = np.concatenate([u.seg_present for u in units])
        self.present = torch.from_numpy(self.present_np).to(self.device)
        images = [u.images for u in units]
        p, g = data.image_tokens, data.image_dim
        self.images = torch.from_numpy(
            np.concatenate(images)
            if sum(len(i) for i in images)
            else np.zeros((0, p, g), np.float32)
        ).to(self.device)
        self.img_kind = torch.from_numpy(
            np.concatenate([u.img_kind for u in units]).astype(np.int64)
        ).to(self.device)
        self.img_readout = torch.from_numpy(
            np.concatenate([u.img_readout for u in units]).astype(np.float32)
        ).to(self.device)


def _pad(arrays: list[np.ndarray], fill, dtype) -> np.ndarray:
    n = max((len(a) for a in arrays), default=0)
    out = np.full((len(arrays), max(n, 1)), fill, dtype=dtype)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


def collate(
    sequences: list[Sequence],
    tensors: FoldTensors,
    config: FitConfig,
    rng: np.random.Generator | None = None,
    n_channels: int = 1,
) -> dict[str, torch.Tensor]:
    """Pad sequences into a batch. With ``rng`` (training) the per-sequence dropouts of §6.8
    are drawn; without it nothing is dropped except modalities the config leaves out."""
    dev = tensors.device
    b = len(sequences)
    seg = [tensors.signal_offset[s.unit] + s.seg for s in sequences]
    img = [tensors.image_offset[s.unit] + s.img for s in sequences]
    lengths = np.array([len(s.seg) for s in sequences])
    batch = {
        "seg": _pad(seg, 0, np.int64),
        "valid": _pad([np.ones(n, bool) for n in lengths], False, bool),
        "tau": _pad([s.tau for s in sequences], 0.0, np.float32),
        "elapsed": _pad([s.elapsed for s in sequences], 0.0, np.float32),
        "entry": _pad([s.entry for s in sequences], 0, np.int64),
        "point": _pad([s.point for s in sequences], False, bool),
        "window": _pad([s.window for s in sequences], False, bool),
        "y": _pad([np.nan_to_num(s.y) for s in sequences], 0.0, np.float32),
        "rul": _pad([np.nan_to_num(s.rul) for s in sequences], 0.0, np.float32),
        "weight": _pad([s.weight for s in sequences], 0.0, np.float32),
        "hi": _pad([np.nan_to_num(s.hi) for s in sequences], 0.0, np.float32),
        "hi_valid": _pad([np.isfinite(s.hi) for s in sequences], False, bool),
        "oracle": _pad([s.oracle for s in sequences], 0.0, np.float32),
        "img": _pad(img, 0, np.int64),
        "img_valid": _pad([np.ones(len(s.img), bool) for s in sequences], False, bool),
        "img_time": _pad([s.img_time for s in sequences], 0.0, np.float32),
        "img_entry": _pad([s.img_entry for s in sequences], 0, np.int64),
    }
    present = tensors.present_np[batch["seg"]] & batch["valid"]  # no GPU round trip (A2)
    drop_signal = np.zeros(b, bool)
    drop_images = np.zeros(b, bool)
    channel_keep = np.ones((b, n_channels), bool)
    token_drop = np.zeros_like(batch["valid"])
    if rng is not None and config.regularize:
        both = rng.random(b) < config.all_modality_dropout
        drop_signal = both | (rng.random(b) < config.modality_dropout)
        drop_images = both | (rng.random(b) < config.modality_dropout)
        channel_keep = rng.random((b, n_channels)) >= config.channel_dropout
        channel_keep[~channel_keep.any(axis=1), 0] = True
        token_drop = rng.random(batch["valid"].shape) < config.token_dropout
        img_drop = rng.random(batch["img_valid"].shape) < config.token_dropout
        batch["img_valid"] &= ~img_drop
    if not config.uses_signal:
        drop_signal[:] = True
    if not config.image_kinds:
        drop_images[:] = True
    batch["img_valid"] &= ~drop_images[:, None]
    batch["present"] = present & ~token_drop & ~drop_signal[:, None]
    batch["channel_keep"] = channel_keep
    batch["drop_signal"] = drop_signal
    batch["drop_images"] = drop_images
    out = {}
    for key, value in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(value))
        out[key] = t.to(dev)
    out["lengths"] = torch.from_numpy(lengths).to(dev)
    return out
