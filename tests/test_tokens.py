"""Tokens and sequences (PROJECT_PLAN.md parts P10, P13)."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.sequences import make_sequence, trajectory_plan
from rulfuse.tokens import build_fold_data, causal_change
from tests.toy_engine import TINY, build_toy


@pytest.fixture(scope="module")
def toy_data(tmp_path_factory):
    t = build_toy(tmp_path_factory.mktemp("toy_tokens"))
    return t, build_fold_data(t.store, t.signal, t.image, t.folds[0], TINY)


def test_causal_change_uses_only_earlier_baseline_rows():
    x = np.array([[1.0], [3.0], [10.0], [20.0]])
    use = np.array([True, True, False, False])
    out = causal_change(x, use)
    assert np.allclose(out[:, 0], [0.0, 1.0, 8.0, 18.0])
    later = x.copy()
    later[3] = 999.0
    assert np.allclose(causal_change(later, use)[:3], out[:3])
    assert np.allclose(causal_change(x, np.zeros(4, bool)), 0.0)


def test_identity_sequence_reproduces_the_store_targets(toy_data):
    t, data = toy_data
    seg = t.store.segments
    for name, unit in data.units.items():
        s = make_sequence(unit)
        rows = t.store.segment_rows(name)
        point = seg["is_prediction_point"][rows].astype(bool)
        assert np.array_equal(s.point, point)
        assert np.allclose(s.tau, seg["tau_s"][rows])
        assert np.allclose(s.y[point], seg["y_s"][rows][point])
        assert np.array_equal(s.window, seg["window"][rows].astype(bool))
        assert np.isclose(s.weight.sum(), 1.0)


def test_statistics_come_from_training_units_only(toy_data):
    t, data = toy_data
    fold = t.folds[0]
    poisoned = type(t.signal)(
        t.signal.folder,
        t.signal.meta,
        np.array(t.signal.embeddings),
        np.array(t.signal.descriptors),
        t.signal.spectral,
        t.signal.done,
    )
    rows = np.isin(t.store.segments["unit"], [fold.test, fold.validation])
    ids = t.store.segments["distinct_segment"][rows]
    poisoned.descriptors[ids[ids >= 0]] = 1e6
    again = build_fold_data(t.store, poisoned, t.image, fold, TINY)
    assert again.transforms["descriptor_mean"] == data.transforms["descriptor_mean"]
    for u in fold.train:
        assert np.allclose(again.units[u].signal, data.units[u].signal)


def test_training_readout_never_uses_its_own_labels(toy_data, monkeypatch):
    t, data = toy_data
    fold = t.folds[0]
    unit = fold.train[0]
    rec = t.store.recordings
    rows = rec["unit"] == unit
    original = rec["hi"].copy()
    rec["hi"] = np.where(rows, rec["hi"] * 3.0 + 50.0, rec["hi"])
    try:
        again = build_fold_data(t.store, t.signal, t.image, fold, TINY)
    finally:
        rec["hi"] = original
    # its standardized readout moves only through the shared HI scale, not its own fit
    mean0, sd0 = data.hi_scale
    mean1, sd1 = again.hi_scale
    raw0 = data.units[unit].img_readout * sd0 + mean0
    raw1 = again.units[unit].img_readout * sd1 + mean1
    assert np.allclose(raw0, raw1, atol=1e-6)


def test_trajectory_augmentation_keeps_baseline_failure_and_images(toy_data):
    t, data = toy_data
    rng = np.random.default_rng(0)
    for unit in data.units.values():
        for _ in range(20):
            plan = trajectory_plan(unit, rng, 0.4)
            base = [p for p in range(len(unit.rec_start)) if unit.rec_baseline[p]]
            assert plan[: len(base)] == base
            assert unit.failure_rec in plan
            s = make_sequence(unit, plan)
            # inside a recording RUL falls by exactly the elapsed time
            for e in np.unique(s.entry):
                m = (s.entry == e) & s.point
                if m.sum() > 1:
                    assert np.allclose(np.diff(s.rul[m]), -np.diff(s.tau[m]))
            assert np.all(np.diff(s.tau) > 0)
            # the failure lies inside the failure recording, at the same offset
            k = plan.index(unit.failure_rec)
            start = s.tau[s.entry == k][0] - unit.seg_elapsed[unit.seg_rec == unit.failure_rec][0]
            assert np.isclose(
                s.failure_s - start, unit.failure_s - unit.rec_start[unit.failure_rec]
            )
            # every image follows its recording: available after its last segment
            for when, e in zip(s.img_time, s.img_entry, strict=True):
                assert when >= s.tau[s.entry == e].max() - 1e-6
            assert len(s.img) == sum(int((unit.img_rec == p).sum()) for p in plan)


def test_duplicates_follow_their_copy_and_shift_later_recordings(toy_data):
    _, data = toy_data
    unit = next(iter(data.units.values()))
    p = int(np.flatnonzero(~unit.rec_baseline)[0])
    plan = list(range(len(unit.rec_start)))
    plan.insert(p + 1, p)
    s = make_sequence(unit, plan)
    d = unit.rec_end[p] - unit.rec_start[p]
    first = s.tau[s.entry == p + 1][0] - unit.seg_elapsed[unit.seg_rec == p][0]
    assert np.isclose(first, unit.rec_end[p])
    assert s.failure_s is not None and unit.failure_s is not None
    assert np.isclose(s.failure_s, unit.failure_s + d)
