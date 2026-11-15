"""Small, physically timed fixtures; no pretrained weights or real experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold, leave_one_unit_out
from rulfuse.core.schema import (
    ArraySignals, ChannelSpec, Dataset, Image, ImageSpec, Recording, Settings, SignalRef, Unit,
)
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import (
    ImageFeatures, SignalFeatures, extract_image_features, extract_signal_features,
)
from rulfuse.features.encoders import PixelStatsEncoder, ProjectionEncoder

MICRO = FitConfig(
    players=("signal", "edge"), d_model=8, rank=4, heads=2, layers=1, ffn=16,
    image_hidden=4, phase_a_steps=2, max_steps=10, eval_every=2, copies=1,
    warmup_steps=2, nll_ramp_steps=2, dropout=0.0, regularize=False,
)


def micro_dataset(root: Path, *, n_units: int = 6, images: bool = True,
                  hi: bool = True, missing: bool = True) -> Dataset:
    """Seven recordings per unit, unequal durations, an operating gap, and post-failure HI."""
    rng = np.random.default_rng(481)
    arrays, units = {}, []
    rate = 64.0
    channels = tuple(ChannelSpec(axis, "acceleration", axis, unit="g") for axis in "xyz")
    for u in range(n_units):
        name = f"U{u + 1}"
        start, records, photos, failure = 0.0, [], [], None
        for r in range(7):
            duration = 5.0 + ((u + r) % 4) * 0.5
            if r == 3:
                start += 2.0 + u / 2
            key = f"{name}-{r}"
            t = np.arange(round(duration * rate)) / rate
            level = 1.0 + 0.1 * r + 0.01 * u
            values = np.stack([
                level * np.sin(2 * np.pi * 3 * t) + 2,
                level * np.cos(2 * np.pi * 5 * t),
                0.1 * t + rng.normal(0, 0.05, len(t)),
            ], axis=1).astype(np.float32)
            arrays[key] = values
            absent = missing and r == 2
            wear = float(20 + 7 * r + u)
            records.append(Recording(
                key, start, duration, None if absent else SignalRef(key, 0, len(values)),
                hi=wear if hi else float("nan"), hi_causal=wear if hi else float("nan"),
                stage=min(3, 1 + r // 3), group=f"g{r}", attrs={"annotation": r},
            ))
            start += duration
            if r == 5:
                failure = start
            if images:
                for kind in ("edge", "texture"):
                    pixels = np.full((8, 12, 3), 35 + 15 * r + u, np.uint8)
                    pixels[::2, ::3, 1] += 20
                    path = root / name / f"{kind}-{r}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    PILImage.fromarray(pixels).save(path)
                    photos.append(Image(key, kind, str(path), start))
        units.append(Unit(name, tuple(records), tuple(photos), failure))
    return Dataset(
        "micro", tuple(units), channels,
        Settings(horizon_s=18.0, sample_rate_hz=rate, model_rate_hz=32.0,
                 baseline_recordings=2, segment_s=2.0, min_remainder_s=1.0,
                 hi_name="wear" if hi else ""),
        ArraySignals(arrays), {k: ImageSpec(8, 12) for k in ("edge", "texture")} if images else {},
        image_root=root,
        provenance={"fixture": 1, "units": n_units, "images": images, "hi": hi, "missing": missing},
    )


@dataclass
class MicroBundle:
    dataset: Dataset
    store: SegmentStore
    signal: SignalFeatures
    image: ImageFeatures | None
    folds: list[Fold]


def build_micro_bundle(root: Path, *, images: bool = True, hi: bool = True) -> MicroBundle:
    ds = micro_dataset(root / "images", images=images, hi=hi)
    store = SegmentStore.open(build_segment_store(ds, root / "store"))
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(dim=4), root=root / "features", log=lambda *_: None,
    )
    image = extract_image_features(
        store, PixelStatsEncoder(), root=root / "features", log=lambda *_: None,
    ) if images else None
    return MicroBundle(ds, store, signal, image, leave_one_unit_out(store.unit_names))
