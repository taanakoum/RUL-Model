"""No model may read a photo that has not been taken yet, whatever the batch (audit F01,
2026-11-15). F0 used to count zero-padded image times as earlier photos, so in a batch of
units with unequal photo counts a unit's early predictions read the image-GRU state of its
later photos. These tests check batch invariance, empty photo sequences, dropped photos and
truncation for both fusion families."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from rulfuse.config import FitConfig
from rulfuse.models.fusion import RULFuse
from rulfuse.sequences import FoldTensors, Sequence, collate

TAU = np.array([5.0, 15.0, 25.0, 35.0])


def _any(x: object) -> Any:
    return x


def _sequence(name: str, times: list[float], n_points: int = len(TAU)) -> Sequence:
    tau = TAU[:n_points]
    rul = 100.0 - tau
    count = len(times)
    return Sequence(
        unit=name,
        seg=np.arange(n_points),
        tau=tau,
        elapsed=tau.copy(),
        entry=np.zeros(n_points, np.int64),
        present=np.ones(n_points, bool),
        point=np.ones(n_points, bool),
        rul=rul,
        y=rul,
        window=np.ones(n_points, bool),
        weight=np.ones(n_points) / n_points,
        hi=np.zeros(n_points),
        oracle=np.zeros(n_points),
        img=np.arange(count),
        img_time=np.asarray(times, float),
        img_entry=np.zeros(count, np.int64),
        failure_s=100.0,
    )


def _setup(fusion: str):
    torch.manual_seed(0)
    cfg = FitConfig(
        players=("signal", "photo"),
        fusion=fusion,
        d_model=16,
        rank=4,
        layers=1,
        heads=2,
        ffn=32,
        dropout=0.0,
        regularize=False,
    )
    units = {}
    for name, count in (("short", 2), ("long", 4), ("none", 0)):
        units[name] = SimpleNamespace(
            signal=np.ones((len(TAU), 1, 2), np.float32),
            seg_present=np.ones(len(TAU), bool),
            images=np.arange(max(count, 1) * 2, dtype=np.float32).reshape(max(count, 1), 1, 2)[
                :count
            ],
            img_kind=np.zeros(count, np.int64),
            img_readout=np.zeros(count, np.float32),
        )
    data = SimpleNamespace(units=units, image_tokens=1, image_dim=2)
    tensors = FoldTensors(_any(data), ["short", "long", "none"], "cpu")
    model = RULFuse(cfg, 100.0, 2, [("sensor", "x", "raw")], 2, 1, 1, False).eval()
    return cfg, tensors, model


def _run(model, cfg, tensors, seqs):
    with torch.no_grad():
        return model(collate(seqs, tensors, cfg, None, 1), tensors)


SHORT = [10.0, 20.0]  # photos at 10 and 20 s
LONG = [10.0, 20.0, 30.0, 40.0]


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_padded_batch_equals_the_unit_alone(fusion):
    cfg, tensors, model = _setup(fusion)
    seqs = [_sequence("short", SHORT), _sequence("long", LONG)]
    batch = collate(seqs, tensors, cfg, None, 1)
    assert batch["img_time"][0].tolist() == [10.0, 20.0, 0.0, 0.0]  # the short unit is padded
    both = _run(model, cfg, tensors, seqs)
    alone = _run(model, cfg, tensors, seqs[:1])
    assert torch.allclose(both["mu"][0], alone["mu"][0], atol=1e-6)
    if fusion == "F0":  # before its first photo (10 s) the short unit has no image expert
        assert not bool(both["experts"]["present"][0, 0, 2])
        assert torch.equal(both["experts"]["present"][0], alone["experts"]["present"][0])


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_a_photo_not_yet_taken_cannot_change_an_earlier_prediction(fusion):
    cfg, tensors, model = _setup(fusion)
    seqs = [_sequence("short", SHORT), _sequence("long", LONG)]
    before = _run(model, cfg, tensors, seqs)
    tensors.images[1] += 50.0  # the short unit's photo taken at 20 s
    after = _run(model, cfg, tensors, seqs)
    assert torch.equal(before["mu"][0, :2], after["mu"][0, :2])  # 5 s and 15 s
    assert not torch.equal(before["mu"][0, 2], after["mu"][0, 2])  # 25 s sees that photo


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_a_unit_without_photos_in_a_batch_with_photos(fusion):
    cfg, tensors, model = _setup(fusion)
    seqs = [_sequence("none", []), _sequence("long", LONG)]
    both = _run(model, cfg, tensors, seqs)
    alone = _run(model, cfg, tensors, seqs[:1])
    assert torch.allclose(both["mu"][0], alone["mu"][0], atol=1e-6)
    if fusion == "F0":
        assert not bool(both["experts"]["present"][0, :, 2].any())


def test_a_dropped_photo_is_as_if_it_was_never_taken():
    """F0: a photo whose token is dropped (valid False) is skipped; the next prediction reads
    the state of the photos before it, like a unit that never had it."""
    cfg, tensors, model = _setup("F0")
    seqs = [_sequence("short", SHORT), _sequence("long", LONG)]
    batch = collate(seqs, tensors, cfg, None, 1)
    batch["img_valid"][0, 1] = False  # drop the short unit's 20 s photo
    with torch.no_grad():
        dropped = model(batch, tensors)
        once = model(collate([_sequence("short", SHORT[:1])], tensors, cfg, None, 1), tensors)
    assert torch.allclose(dropped["mu"][0], once["mu"][0], atol=1e-6)


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_truncation_leaves_earlier_predictions_unchanged(fusion):
    cfg, tensors, model = _setup(fusion)
    full = _run(model, cfg, tensors, [_sequence("short", SHORT), _sequence("long", LONG)])
    cut = _run(model, cfg, tensors, [_sequence("short", SHORT, 2), _sequence("long", LONG)])
    assert torch.allclose(full["mu"][0, :2], cut["mu"][0, :2], atol=1e-6)
