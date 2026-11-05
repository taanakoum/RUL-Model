"""Inputs of OmniFuser-RUL (docs/OMNIFUSER_RUL_PLAN.md §4, §6).

One OmniFuser time step is one segment of the store. A sample is the ``INPUT_STEPS``
segments ending at a prediction point, with that point's capped target y and point weight.

Per step the force vector holds, in this order (``Layout``):

* the 7 level descriptors of every model channel (``channels.descriptors``), standardized
  on the fold's training units; zeros where the signal is missing (D8) or padded;
* the signal-present flag;
* the usage inputs log(1 + τ/H) and e/100 s, standardized on the training units.

Per step and image type, the photo is the latest one with ``available_s < τ`` (D9, the
store's ``image_visible`` rule). The frozen EfficientNet-B0 features of the types in use
are put side by side into one composite vector per distinct combination; a step with no
visible photo yet points at an all-zero vector.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from rulfuse import paths
from rulfuse.core.channels import DESCRIPTOR_NAMES, DESCRIPTORS_VERSION, descriptors, model_channels
from rulfuse.core.images import normalize, prepared_pixels
from rulfuse.core.schema import TIME_TOLERANCE_S, SignalSource
from rulfuse.core.store import SegmentStore, stable_hash

INPUT_STEPS = 96
FEATURES_VERSION = 1
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

    key = {
        "store": store.signature,
        "descriptors": DESCRIPTORS_VERSION,
        "version": FEATURES_VERSION,
    }
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
        if count % 500 == 0:
            log(f"[omnifuser-rul] descriptors {count}/{len(order)}")
    _save(folder, out, {"key": key, "shape": list(out.shape)})
    return out


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
    (tests); it must have ``backbone_features`` and the same cache key is not reused."""
    import torch
    from omnifuser.models import ImageFeatureEncoder

    specs = {kind: vars(spec) for kind, spec in sorted(store.image_specs.items())}
    key = {
        "store": store.signature,
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
    point: np.ndarray  # prediction points
    y_s: np.ndarray  # capped target (nan off points)
    rul_s: np.ndarray
    window: np.ndarray
    weight: np.ndarray  # point weights, summing to 1 over the unit's points

    @property
    def point_index(self) -> np.ndarray:
        return np.flatnonzero(self.point)


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
    store: SegmentStore, descriptor_values: np.ndarray, kinds: Sequence[str]
) -> dict[str, UnitSteps]:
    """Per unit, the step table of every segment (in operating-time order)."""
    seg = store.segments
    horizon = store.settings.horizon_s
    n_desc = descriptor_values.shape[1] * descriptor_values.shape[2]
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
        rul = seg["rul_s"][rows].astype(np.float64)
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
            rul_s=rul,
            window=seg["window"][rows].astype(bool),
            weight=point_weights(np.nan_to_num(rul, nan=np.inf), point, horizon),
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


class Windows(Dataset):
    """Samples of some units: one per prediction point."""

    def __init__(
        self,
        units: Sequence[UnitSteps],
        standardizer: Standardizer,
        image_features: np.ndarray,
        kinds_used: Sequence[int],
        input_steps: int = INPUT_STEPS,
    ) -> None:
        self.units = list(units)
        self.matrices = [standardizer.matrix(u) for u in self.units]
        self.features = image_features
        self.kinds_used = list(kinds_used)  # columns of UnitSteps.image_rows in use
        self.input_steps = input_steps
        self.index = [(i, int(p)) for i, u in enumerate(self.units) for p in u.point_index]
        self.view_width = image_features.shape[1] if len(image_features) else IMAGE_FEATURE_WIDTH

    def __len__(self) -> int:
        return len(self.index)

    @property
    def composite_width(self) -> int:
        return self.view_width * max(len(self.kinds_used), 1)

    def __getitem__(self, item: int) -> dict:
        u, p = self.index[item]
        unit, matrix = self.units[u], self.matrices[u]
        start = p - self.input_steps + 1
        force = np.zeros((self.input_steps, matrix.shape[1]), np.float32)
        keys = np.full((self.input_steps, max(len(self.kinds_used), 1)), -1, dtype=np.int64)
        lo = max(start, 0)
        force[lo - start :] = matrix[lo : p + 1]
        if self.kinds_used:
            keys[lo - start :] = unit.image_rows[lo : p + 1][:, self.kinds_used]
        # distinct composites in the window; slot 0 is the all-zero "no photo yet"
        combos, inverse = np.unique(keys, axis=0, return_inverse=True)
        composites = [np.zeros(self.composite_width, np.float32)]
        slot_of = []
        for combo in combos:
            if (combo < 0).all() or not self.kinds_used:
                slot_of.append(0)
                continue
            parts = [
                self.features[r] if r >= 0 else np.zeros(self.view_width, np.float32) for r in combo
            ]
            composites.append(np.concatenate(parts).astype(np.float32))
            slot_of.append(len(composites) - 1)
        indices = np.asarray(slot_of, dtype=np.int64)[inverse.reshape(-1)]
        return {
            "force": force,
            "images": np.stack(composites),
            "image_indices": indices,
            "y": np.float32(unit.y_s[p]),
            "weight": np.float32(unit.weight[p]),
            "unit": u,
            "row": int(unit.rows[p]),
        }


def collate(samples: Sequence[dict]):
    """Batch: images padded to the largest composite count (padding is never indexed)."""
    import torch

    most = max(len(s["images"]) for s in samples)
    width = samples[0]["images"].shape[1]
    images = np.zeros((len(samples), most, width), np.float32)
    for i, s in enumerate(samples):
        images[i, : len(s["images"])] = s["images"]
    return {
        "force": torch.from_numpy(np.stack([s["force"] for s in samples])),
        "images": torch.from_numpy(images),
        "image_indices": torch.from_numpy(np.stack([s["image_indices"] for s in samples])),
        "y": torch.tensor([s["y"] for s in samples]),
        "weight": torch.tensor([s["weight"] for s in samples]),
        "unit": torch.tensor([s["unit"] for s in samples]),
        "row": torch.tensor([s["row"] for s in samples]),
    }
