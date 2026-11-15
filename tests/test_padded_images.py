"""F0 must not read a photo that has not been taken yet when a batch pads image times
(audit F01, 2026-11-15): a unit's prediction in a padded batch equals its prediction alone."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from rulfuse.config import FitConfig
from rulfuse.models.fusion import RULFuse
from rulfuse.sequences import FoldTensors, Sequence, collate


def Any_(x: object) -> Any:
    return x


def _sequence(name: str, count: int) -> Sequence:
    return Sequence(
        unit=name,
        seg=np.arange(3),
        tau=np.array([5.0, 15.0, 25.0]),
        elapsed=np.array([5.0, 15.0, 25.0]),
        entry=np.zeros(3, np.int64),
        present=np.ones(3, bool),
        point=np.ones(3, bool),
        rul=np.array([95.0, 85.0, 75.0]),
        y=np.array([95.0, 85.0, 75.0]),
        window=np.ones(3, bool),
        weight=np.ones(3) / 3,
        hi=np.zeros(3),
        oracle=np.zeros(3),
        img=np.arange(count),
        img_time=np.arange(1, count + 1) * 10.0,  # photos at 10, 20, ... s
        img_entry=np.zeros(count, np.int64),
        failure_s=100.0,
    )


def _setup():
    torch.manual_seed(0)
    cfg = FitConfig(
        players=("signal", "photo"),
        fusion="F0",
        d_model=16,
        rank=4,
        layers=1,
        heads=2,
        ffn=32,
        dropout=0.0,
        regularize=False,
    )
    units = {}
    for name, count in (("short", 2), ("long", 4)):
        units[name] = SimpleNamespace(
            signal=np.ones((3, 1, 2), np.float32),
            seg_present=np.ones(3, bool),
            images=np.arange(count * 2, dtype=np.float32).reshape(count, 1, 2),
            img_kind=np.zeros(count, np.int64),
            img_readout=np.zeros(count, np.float32),
        )
    data = SimpleNamespace(units=units, image_tokens=1, image_dim=2)
    tensors = FoldTensors(Any_(data), ["short", "long"], "cpu")
    model = RULFuse(cfg, 100.0, 2, [("sensor", "x", "raw")], 2, 1, 1, False).eval()
    seqs = [_sequence("short", 2), _sequence("long", 4)]
    return cfg, tensors, model, seqs


def test_padded_batch_equals_the_unit_alone():
    cfg, tensors, model, seqs = _setup()
    batch = collate(seqs, tensors, cfg, None, 1)
    alone = collate(seqs[:1], tensors, cfg, None, 1)
    assert batch["img_time"][0].tolist() == [10.0, 20.0, 0.0, 0.0]  # the short unit is padded
    with torch.no_grad():
        both, single = model(batch, tensors), model(alone, tensors)
    # before its first photo (10 s) the short unit has no image expert, padded or not
    assert not bool(both["experts"]["present"][0, 0, 2])
    assert torch.allclose(both["mu"][0], single["mu"][0], atol=1e-6)
    assert torch.allclose(both["experts"]["present"][0], single["experts"]["present"][0])


def test_a_photo_not_yet_taken_cannot_change_an_earlier_prediction():
    cfg, tensors, model, seqs = _setup()
    batch = collate(seqs, tensors, cfg, None, 1)
    with torch.no_grad():
        before = model(batch, tensors)
        tensors.images[1] += 50.0  # the short unit's photo taken at 20 s
        after = model(batch, tensors)
    assert torch.equal(before["mu"][0, 0], after["mu"][0, 0])  # the 5 s prediction
    assert torch.equal(before["mu"][0, 1], after["mu"][0, 1])  # the 15 s prediction
    assert not torch.equal(before["mu"][0, 2], after["mu"][0, 2])  # 25 s sees that photo
