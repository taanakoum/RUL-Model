"""A small toy dataset with features, and a tiny fit configuration, for the engine tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rulfuse.adapters import toy
from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold, leave_one_unit_out
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import (
    ImageFeatures,
    SignalFeatures,
    extract_image_features,
    extract_signal_features,
)
from rulfuse.features.encoders import PixelStatsEncoder, ProjectionEncoder

TINY = FitConfig(
    players=("signal", "surface"),
    d_model=16,
    rank=4,
    ffn=32,
    heads=2,
    layers=1,
    image_hidden=8,
    phase_a_steps=4,
    max_steps=24,
    eval_every=5,
    copies=1,
    warmup_steps=2,
    nll_ramp_steps=5,
)


@dataclass
class Toy:
    store: SegmentStore
    signal: SignalFeatures
    image: ImageFeatures
    folds: list[Fold]


def build_toy(root: Path, n_units: int = 5) -> Toy:
    ds = toy.build(root / "images", n_units=n_units)
    store = SegmentStore.open(build_segment_store(ds, root / "store"))
    quiet = lambda *_: None  # noqa: E731
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=root / "f", log=quiet
    )
    image = extract_image_features(store, PixelStatsEncoder(), root=root / "f", log=quiet)
    return Toy(store, signal, image, leave_one_unit_out(store.unit_names))
