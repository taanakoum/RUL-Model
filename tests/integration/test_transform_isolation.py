"""P7/P9/P10/P13: poison real token-building inputs and check mathematical invariants."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.probe import cross_fitted_readout
from rulfuse.sequences import make_sequence, trajectory_plan
from rulfuse.tokens import build_fold_data
from tests.factories import MICRO

pytestmark = [pytest.mark.integration,
              pytest.mark.requirements("P3", "P5", "P7", "P9", "P10", "P13", "D14", "D18")]


@pytest.mark.adversarial
def test_heldout_descriptor_and_health_poison_cannot_change_training_inputs(micro_bundle):
    b = micro_bundle
    fold = b.folds[0]
    original = build_fold_data(b.store, b.signal, b.image, fold, MICRO)
    store = copy.deepcopy(b.store)
    signals = copy.copy(b.signal)
    signals.descriptors = np.array(signals.descriptors)
    held = np.isin(store.segments["unit"], [fold.test, fold.validation])
    ids = store.segments["distinct_segment"][held].astype(int)
    signals.descriptors[ids[ids >= 0]] = 1e8
    for table in (store.recordings, store.segments):
        mine = np.isin(table["unit"], [fold.test, fold.validation])
        table["hi"][mine] = 1e8
    changed = build_fold_data(store, signals, b.image, fold, MICRO)
    assert original.transforms == changed.transforms
    assert original.hi_scale == changed.hi_scale
    for name in fold.train:
        np.testing.assert_array_equal(original.units[name].signal, changed.units[name].signal)
        np.testing.assert_array_equal(original.units[name].img_readout, changed.units[name].img_readout)


@pytest.mark.adversarial
def test_cross_fitting_excludes_own_labels_including_hyperparameter_selection():
    rng = np.random.default_rng(92)
    groups = np.repeat(["A", "B", "C", "V", "T"], 6)
    x = rng.normal(size=(len(groups), 5))
    y = 3 * x[:, 0] + rng.normal(size=len(groups))
    base = cross_fitted_readout(x, y, groups, ["A", "B", "C"])
    poison = y.copy()
    poison[groups == "A"] = rng.normal(0, 1e6, 6)
    own = cross_fitted_readout(x, poison, groups, ["A", "B", "C"])
    np.testing.assert_array_equal(base[groups == "A"], own[groups == "A"])
    poison = y.copy()
    poison[np.isin(groups, ["V", "T"])] = 1e8
    unseen = cross_fitted_readout(x, poison, groups, ["A", "B", "C"])
    np.testing.assert_array_equal(base, unseen)


@pytest.mark.adversarial
def test_adapter_annotations_and_stage_codes_never_enter_tokens(micro_bundle, tmp_path):
    b = micro_bundle
    changed_units = tuple(replace(u, recordings=tuple(
        replace(r, stage=999, group="SECRET_TARGET", attrs={"target": 1e15}) for r in u.recordings
    ), attrs={"unit_identity": "SECRET_TARGET"}) for u in b.dataset.units)
    ds = replace(b.dataset, units=changed_units)
    store = SegmentStore.open(build_segment_store(ds, tmp_path / "annotations"))
    a = build_fold_data(b.store, b.signal, b.image, b.folds[0], MICRO)
    c = build_fold_data(store, b.signal, b.image, b.folds[0], MICRO)
    for name in a.units:
        np.testing.assert_array_equal(a.units[name].signal, c.units[name].signal)
        np.testing.assert_array_equal(a.units[name].images, c.units[name].images)
        np.testing.assert_array_equal(a.units[name].img_readout, c.units[name].img_readout)


@pytest.mark.parametrize("seed", range(10))
def test_augmented_trajectories_preserve_baseline_failure_photos_and_exact_labels(micro_bundle, seed):
    b = micro_bundle
    data = build_fold_data(b.store, b.signal, b.image, b.folds[0], MICRO)
    for unit in data.units.values():
        plan = trajectory_plan(unit, np.random.default_rng(seed), .45)
        baseline = np.flatnonzero(unit.rec_baseline).tolist()
        assert all(plan.count(p) == 1 for p in baseline)
        assert plan.count(unit.failure_rec) == 1
        seq = make_sequence(unit, plan, MICRO.w_far)
        assert np.all(np.diff(seq.tau) > 0)
        assert seq.failure_s is not None
        np.testing.assert_allclose(seq.rul[seq.point], seq.failure_s - seq.tau[seq.point], atol=1e-10)
        np.testing.assert_allclose(seq.y[seq.point], np.minimum(seq.rul[seq.point], unit.horizon_s))
        assert seq.weight.sum() == pytest.approx(1.)
        assert not seq.weight[~seq.point].any()
        for entry, recording in enumerate(plan):
            ids = seq.img[seq.img_entry == entry]
            np.testing.assert_array_equal(np.sort(ids), unit.rec_images[recording])
            if len(ids):
                signal_rows = np.flatnonzero(seq.entry == entry)
                assert (seq.img_time[seq.img_entry == entry] >= seq.tau[signal_rows[-1]]).all()


def test_deleting_recordings_keeps_unrecorded_operating_gaps(micro_bundle):
    b = micro_bundle
    unit = build_fold_data(b.store, b.signal, b.image, b.folds[0], MICRO).units[b.folds[0].test]
    plan = [0, 1, 3, 4, 5, 6]  # remove recording 2, keep the physical gap before recording 3
    seq = make_sequence(unit, plan)
    last = seq.tau[np.flatnonzero(seq.entry == 1)[-1]]
    first = seq.tau[np.flatnonzero(seq.entry == 2)[0]]
    original_gap = unit.rec_start[3] - unit.rec_end[2]
    assert first - unit.seg_elapsed[seq.seg[np.flatnonzero(seq.entry == 2)[0]]] - last \
        == pytest.approx(original_gap)
