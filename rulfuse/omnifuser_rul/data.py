"""Inputs of OmniFuser-RUL (docs/OMNIFUSER_RUL_PLAN.md §4, §6; version 3).

One OmniFuser time step is one segment of the store. B6 uses its own store cut at
``DEFAULT_SEGMENT_S`` = 1 s, OmniFuser's native step (O7): 96 steps are 96 s, about one
pass, and a fold trains on about 29,000 prediction points instead of 3,000 at 10 s. It is
scored on the points of the project's 10 s grid (``EVAL_GRID_S``, D7), which a 1 s store
contains exactly, with the same τ and RUL.

Per step the force vector holds, in this order (``Layout``):

* the 7 level descriptors of every model channel (``channels.descriptors``), standardized
  on the fold's training units; zeros where the signal is missing (D8) or padded;
* the signal-present flag;
* the usage inputs log(1 + τ/H) and e/100 s, standardized on the training units.

Per step and image type, the photo is the latest one with ``available_s < τ`` (D9). The
frozen EfficientNet-B0 features of the types in use are put side by side into one
composite vector per distinct combination; ``UnitInputs`` keeps one table of composites per
unit (row 0 is the all-zero "no photo yet") and, per step, the row to read. A window is the
96 steps ending at a point; it reads the unit's table through those per-step rows, which is
OmniFuser's ``(images, image_indices)`` interface with precomputed features.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rulfuse import paths
from rulfuse.core.channels import DESCRIPTOR_NAMES, DESCRIPTORS_VERSION, descriptors, model_channels
from rulfuse.core.images import normalize, prepared_pixels
from rulfuse.core.schema import TIME_TOLERANCE_S, SignalSource
from rulfuse.core.store import SegmentStore, grid_mask, stable_hash

INPUT_STEPS = 96
DEFAULT_SEGMENT_S = 1.0  # O7: OmniFuser's native step
EVAL_GRID_S = 10.0  # D7: the project's prediction points, shared with RUL-Fuse and B0-B5
FEATURES_VERSION = 2
IMAGE_ENCODER = "efficientnet_b0-imagenet1k_v1"
IMAGE_FEATURE_WIDTH = 1280
FAR_WEIGHT = 0.3  # PROJECT_PLAN.md §7.3: points with RUL > H
USAGE_NAMES = ("log1p_tau_over_h", "elapsed_over_100s")


def features_dir(store: SegmentStore, name: str, key: dict, root: Path | None = None) -> Path:
    base = root if root is not None else paths.interim_dir() / "features"
    return base / store.dataset / f"{name}-{stable_hash(key)}"


def _save(folder: Path, array: np.ndarray, meta: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "values.npy.tmp"
    with tmp.open("wb") as file:
        np.save(file, array)
    tmp.replace(folder / "values.npy")
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def _load(folder: Path, key: dict) -> np.ndarray | None:
    meta_path = folder / "meta.json"
    if not (folder / "values.npy").is_file() or not meta_path.is_file():
        return None
    # compared in JSON form: a tuple in the key (e.g. a photo crop) reads back as a list
    if json.loads(meta_path.read_text(encoding="utf-8")).get("key") != json.loads(json.dumps(key)):
        raise ValueError(f"{folder}: cache key mismatch")
    return np.load(folder / "values.npy")


def descriptor_cache(
    store: SegmentStore, signals: SignalSource, log=print, root: Path | None = None
) -> np.ndarray:
    """(distinct segments, model channels, 7) level descriptors, computed once per store with
    the same reading, resampling and drift reference as the feature caches."""
    from rulfuse.features.cache import _segment_inputs, distinct_rows

    key = {"store": store.signature, "descriptors": DESCRIPTORS_VERSION, "version": 1}
    folder = features_dir(store, "omnifuser-rul-descriptors", key, root)
    cached = _load(folder, key)
    if cached is not None:
        return cached
    first = distinct_rows(store)
    out = np.zeros(
        (len(first), len(model_channels(store.channels)), len(DESCRIPTOR_NAMES)), np.float32
    )
    order = np.argsort(store.segments["source"][first], kind="stable")
    reader: dict = {}
    for count, d in enumerate(order, start=1):
        segment, head = _segment_inputs(store, signals, int(first[d]), reader)
        out[d] = descriptors(segment, store.settings.model_rate_hz, head)
        if count % 5000 == 0:
            log(f"[omnifuser-rul] descriptors {count}/{len(order)}")
    _save(folder, out, {"key": key, "shape": list(out.shape)})
    return out


def image_table_key(store: SegmentStore) -> str:
    """A hash of the store's image table (unit, type, key, path, availability). It does not
    depend on the segment length, so stores cut at 1 s and 10 s share the photo features."""
    img = store.images
    rows = [
        [str(u), str(k), str(key), str(p), round(float(a), 6)]
        for u, k, key, p, a in zip(
            img["unit"], img["kind"], img["key"], img["path"], img["available_s"], strict=True
        )
    ]
    return stable_hash({"dataset": store.dataset, "images": rows})


def image_feature_cache(
    store: SegmentStore,
    device: str = "cpu",
    batch: int = 16,
    log=print,
    encoder=None,
    root: Path | None = None,
) -> np.ndarray:
    """(images, 1280) pooled features of the frozen ImageNet EfficientNet-B0 (OmniFuser's
    ``ImageFeatureEncoder.backbone_features``), one row per row of the store's image table,
    each image prepared with its type's ``ImageSpec``. ``encoder`` replaces the network
    (tests); it must have ``backbone_features``, and its cache key differs."""
    import torch
    from omnifuser.models import ImageFeatureEncoder

    specs = {kind: vars(spec) for kind, spec in sorted(store.image_specs.items())}
    key = {
        "images": image_table_key(store),
        "encoder": IMAGE_ENCODER if encoder is None else type(encoder).__name__,
        "specs": specs,
        "version": FEATURES_VERSION,
    }
    folder = features_dir(store, "omnifuser-rul-images", key, root)
    cached = _load(folder, key)
    if cached is not None:
        return cached
    if encoder is None:
        encoder = ImageFeatureEncoder("efficientnet_b0", 32, pretrained=True, freeze_weights=True)
    encoder = encoder.to(device).eval()
    kinds = store.images["kind"]
    out = np.zeros((len(kinds), IMAGE_FEATURE_WIDTH), np.float32)
    done = 0
    for kind, spec in store.image_specs.items():
        rows = store.image_rows(kind=kind)
        for start in range(0, len(rows), batch):
            part = rows[start : start + batch]
            pixels = np.stack(
                [normalize(prepared_pixels(store.image_path(int(r)), spec)) for r in part]
            )
            with torch.no_grad():
                out[part] = (
                    encoder.backbone_features(torch.from_numpy(pixels).to(device)).cpu().numpy()
                )
            done += len(part)
            if done % (batch * 20) < batch:
                log(f"[omnifuser-rul] image features {done}/{len(kinds)}")
    _save(folder, out, {"key": key, "shape": list(out.shape)})
    return out


@dataclass(frozen=True)
class Layout:
    """Column layout of the per-step force vector."""

    descriptors: int  # model channels x 7

    @property
    def flag(self) -> int:
        return self.descriptors

    @property
    def usage(self) -> slice:
        return slice(self.descriptors + 1, self.descriptors + 1 + len(USAGE_NAMES))

    @property
    def width(self) -> int:
        return self.descriptors + 1 + len(USAGE_NAMES)


@dataclass
class UnitSteps:
    """One unit's segments in time order, with everything a window needs."""

    name: str
    rows: np.ndarray  # segment rows of the store
    descriptors: np.ndarray  # (n, model channels x 7), zeros where the signal is missing
    present: np.ndarray  # (n,) bool
    usage: np.ndarray  # (n, 2) raw usage inputs
    image_rows: np.ndarray  # (n, image types) latest visible image row per type, -1 if none
    tau_s: np.ndarray
    point: np.ndarray  # prediction points (a prefix of the steps)
    y_s: np.ndarray  # capped target (nan off points)
    rul_s: np.ndarray
    window: np.ndarray
    weight: np.ndarray  # point weights over all points, summing to 1
    grid: np.ndarray  # points on the evaluation grid
    grid_weight: np.ndarray  # point weights over the grid points only, summing to 1

    @property
    def point_index(self) -> np.ndarray:
        return np.flatnonzero(self.point)

    @property
    def n_points(self) -> int:
        return int(self.point.sum())


def point_weights(rul_s: np.ndarray, point: np.ndarray, horizon_s: float) -> np.ndarray:
    """PROJECT_PLAN.md §7.3: 1 inside the window, FAR_WEIGHT beyond it, normalized per unit."""
    v = np.where(rul_s <= horizon_s + TIME_TOLERANCE_S, 1.0, FAR_WEIGHT) * point
    total = v.sum()
    return v / total if total > 0 else v


def latest_visible(available_s: np.ndarray, rows: np.ndarray, tau_s: np.ndarray) -> np.ndarray:
    """For each τ, the row of the latest image with ``available_s < τ`` (-1 if none)."""
    order = np.argsort(available_s, kind="stable")
    sorted_available = available_s[order]
    position = np.searchsorted(sorted_available, tau_s - TIME_TOLERANCE_S, side="left") - 1
    out = np.full(len(tau_s), -1, dtype=np.int64)
    seen = position >= 0
    out[seen] = rows[order[position[seen]]]
    return out


def build_steps(
    store: SegmentStore,
    descriptor_values: np.ndarray,
    kinds: Sequence[str],
    eval_grid_s: float = EVAL_GRID_S,
) -> dict[str, UnitSteps]:
    """Per unit, the step table of every segment (in operating-time order)."""
    seg = store.segments
    horizon = store.settings.horizon_s
    n_desc = descriptor_values.shape[1] * descriptor_values.shape[2]
    on_grid = grid_mask(store, eval_grid_s)
    out: dict[str, UnitSteps] = {}
    for unit in store.unit_names:
        rows = store.segment_rows(unit)
        rows = rows[np.argsort(seg["tau_s"][rows], kind="stable")]
        tau = seg["tau_s"][rows].astype(np.float64)
        present = seg["signal_present"][rows].astype(bool)
        distinct = seg["distinct_segment"][rows].astype(np.int64)
        desc = np.zeros((len(rows), n_desc), np.float32)
        have = present & (distinct >= 0)
        desc[have] = descriptor_values[distinct[have]].reshape(int(have.sum()), n_desc)
        usage = np.stack(
            [np.log1p(tau / horizon), seg["elapsed_s"][rows].astype(np.float64) / 100.0], axis=1
        ).astype(np.float32)
        image_rows = np.full((len(rows), len(kinds)), -1, dtype=np.int64)
        for k, kind in enumerate(kinds):
            mine = store.image_rows(unit=unit, kind=kind)
            if len(mine):
                image_rows[:, k] = latest_visible(
                    store.images["available_s"][mine].astype(np.float64), mine, tau
                )
        point = seg["is_prediction_point"][rows].astype(bool)
        if point.any() and not point[: int(point.sum())].all():
            raise ValueError(f"{unit}: prediction points must be a prefix of the steps")
        rul = np.nan_to_num(seg["rul_s"][rows].astype(np.float64), nan=np.inf)
        grid = point & on_grid[rows]
        out[unit] = UnitSteps(
            name=unit,
            rows=rows,
            descriptors=desc,
            present=present,
            usage=usage,
            image_rows=image_rows,
            tau_s=tau,
            point=point,
            y_s=seg["y_s"][rows].astype(np.float64),
            rul_s=seg["rul_s"][rows].astype(np.float64),
            window=seg["window"][rows].astype(bool),
            weight=point_weights(rul, point, horizon),
            grid=grid,
            grid_weight=point_weights(rul, grid, horizon),
        )
    return out


@dataclass(frozen=True)
class Standardizer:
    """Mean and spread of the descriptor and usage columns over the training units."""

    desc_mean: np.ndarray
    desc_sd: np.ndarray
    usage_mean: np.ndarray
    usage_sd: np.ndarray

    @classmethod
    def fit(cls, units: Sequence[UnitSteps]) -> Standardizer:
        desc = np.concatenate([u.descriptors[u.present] for u in units])
        usage = np.concatenate([u.usage for u in units])
        return cls(
            desc.mean(axis=0),
            np.maximum(desc.std(axis=0), 1e-6),
            usage.mean(axis=0),
            np.maximum(usage.std(axis=0), 1e-6),
        )

    def matrix(self, unit: UnitSteps) -> np.ndarray:
        """(n, Layout.width) force vectors of one unit."""
        desc = (unit.descriptors - self.desc_mean) / self.desc_sd
        desc[~unit.present] = 0.0
        usage = (unit.usage - self.usage_mean) / self.usage_sd
        return np.concatenate(
            [desc, unit.present[:, None].astype(np.float32), usage], axis=1
        ).astype(np.float32)


@dataclass
class UnitInputs:
    """One unit's model inputs for a fit, up to its last prediction point."""

    steps: UnitSteps
    force: np.ndarray  # (n, width) standardized force vectors
    composites: np.ndarray  # (m, views x 1280); row 0 all zero ("no photo yet")
    comp_index: np.ndarray  # (n,) composite row of each step

    @property
    def name(self) -> str:
        return self.steps.name

    @property
    def n(self) -> int:
        return len(self.force)


def unit_inputs(
    unit: UnitSteps,
    standardizer: Standardizer,
    image_features: np.ndarray,
    kinds_used: Sequence[int],
) -> UnitInputs:
    n = unit.n_points
    force = standardizer.matrix(unit)[:n]
    views = max(len(kinds_used), 1)
    width = image_features.shape[1] if len(image_features) else IMAGE_FEATURE_WIDTH
    if not kinds_used:
        return UnitInputs(
            unit, force, np.zeros((1, width * views), np.float32), np.zeros(n, np.int64)
        )
    keys = unit.image_rows[:n][:, list(kinds_used)]
    combos, inverse = np.unique(keys, axis=0, return_inverse=True)
    composites = [np.zeros(width * views, np.float32)]
    slot_of = []
    for combo in combos:
        if (combo < 0).all():
            slot_of.append(0)
            continue
        parts = [image_features[r] if r >= 0 else np.zeros(width, np.float32) for r in combo]
        composites.append(np.concatenate(parts).astype(np.float32))
        slot_of.append(len(composites) - 1)
    index = np.asarray(slot_of, dtype=np.int64)[np.asarray(inverse).reshape(-1)]
    return UnitInputs(unit, force, np.stack(composites), index)


def sample_pairs(
    units: Sequence[UnitInputs], n_pairs: int, max_lag: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_pairs, 3) rows of (unit, earlier step, later step). The later step is a uniformly
    drawn prediction point of the units; the earlier one is ``lag`` steps before it, with
    lag 1 (the previous segment) half of the time and uniform on 2..max_lag otherwise, cut
    at the unit's first step. A pair whose later step is the first step has earlier = later
    (no temporal pair; its targets still count)."""
    sizes = np.array([u.n for u in units])
    flat = rng.integers(0, sizes.sum(), size=n_pairs)
    unit = np.searchsorted(np.cumsum(sizes), flat, side="right")
    later = flat - np.concatenate([[0], np.cumsum(sizes)[:-1]])[unit]
    lag = np.where(rng.random(n_pairs) < 0.5, 1, rng.integers(2, max(max_lag, 2) + 1, n_pairs))
    earlier = np.maximum(later - lag, 0)
    return np.stack([unit, earlier, later], axis=1).astype(np.int64)


class FitTensors:
    """The inputs of some units on a device, padded so that windows and memory read them
    with tensor indexing: force and per-step composite rows padded at the front with
    ``input_steps - 1`` empty steps (windows) and at the end to the longest unit (memory)."""

    def __init__(self, units: Sequence[UnitInputs], input_steps: int, device: str) -> None:
        import torch

        self.units = list(units)
        self.input_steps = input_steps
        front = input_steps - 1
        n_max = max(u.n for u in self.units)
        m_max = max(len(u.composites) for u in self.units)
        width = self.units[0].force.shape[1]
        comp_width = self.units[0].composites.shape[1]
        force = np.zeros((len(self.units), front + n_max, width), np.float32)
        index = np.zeros((len(self.units), front + n_max), np.int64)
        composites = np.zeros((len(self.units), m_max, comp_width), np.float32)
        for i, u in enumerate(self.units):
            force[i, front : front + u.n] = u.force
            index[i, front : front + u.n] = u.comp_index
            composites[i, : len(u.composites)] = u.composites
        self.force = torch.from_numpy(force).to(device)
        self.index = torch.from_numpy(index).to(device)
        self.composites = torch.from_numpy(composites).to(device)
        self.lengths = torch.tensor([u.n for u in self.units], device=device)
        self.offsets = torch.arange(input_steps, device=device)

    def memory_inputs(self):
        """(force (U, n_max, width), composites (U, m, F), comp rows (U, n_max), lengths)."""
        front = self.input_steps - 1
        return self.force[:, front:], self.composites, self.index[:, front:], self.lengths

    def windows(self, unit, step):
        """Tensors (B,) of unit and step -> force (B, T, width), images (B, m, F) and image
        indices (B, T): the T steps ending at each step (front padding: zeros and row 0)."""
        positions = step[:, None] + self.offsets[None]  # padded coordinates
        force = self.force[unit[:, None], positions]
        image_indices = self.index[unit[:, None], positions]
        return force, self.composites[unit], image_indices
