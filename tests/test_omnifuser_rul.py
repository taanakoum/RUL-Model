"""OmniFuser-RUL, baseline B6 (docs/OMNIFUSER_RUL_PLAN.md §9): the checks V-1 to V-10 on
the toy dataset, and V-3 on the real store."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from rulfuse.adapters import toy
from rulfuse.baselines import score_usage_baselines
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.schema import TIME_TOLERANCE_S
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import extract_signal_features
from rulfuse.features.encoders import ProjectionEncoder
from rulfuse.omnifuser_rul import aggregate, data
from rulfuse.omnifuser_rul.model import OmniFuserRUL
from rulfuse.omnifuser_rul.study import modality_arms
from rulfuse.omnifuser_rul.train import ArmSpec, arm_dir, run_arm, select_epoch

KINDS = ["surface"]


class FakePhotoEncoder(nn.Module):
    """1,280 features that follow image brightness (no download)."""

    def backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return mean * torch.linspace(0.5, 1.5, data.IMAGE_FEATURE_WIDTH)[None]


@pytest.fixture(scope="module")
def toy_b6(tmp_path_factory):
    root = tmp_path_factory.mktemp("b6")
    ds = toy.build(root / "images", n_units=5)
    store = SegmentStore.open(build_segment_store(ds, root / "store"))
    descriptors = data.descriptor_cache(store, ds.signals, log=lambda *_: None, root=root / "f")
    features = data.image_feature_cache(
        store, encoder=FakePhotoEncoder(), log=lambda *_: None, root=root / "f"
    )
    steps = data.build_steps(store, descriptors, KINDS)
    return ds, store, descriptors, features, steps, root


def _windows(steps, features, units, kinds_used=(0,)):
    standardizer = data.Standardizer.fit([steps[u] for u in units])
    return data.Windows([steps[u] for u in units], standardizer, features, list(kinds_used))


def test_descriptors_are_the_feature_caches_descriptors(toy_b6, tmp_path):
    ds, store, descriptors, *_ = toy_b6
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=tmp_path, log=lambda *_: None
    )
    np.testing.assert_allclose(descriptors, np.asarray(signal.descriptors), rtol=1e-6)
    _, _, _, _, _, root = toy_b6
    again = data.descriptor_cache(store, ds.signals, log=lambda *_: None, root=root / "f")
    assert again is not descriptors  # read back from the cache
    np.testing.assert_array_equal(again, descriptors)


def test_steps_follow_the_store_targets_and_weights(toy_b6):
    _, store, _, _, steps, _ = toy_b6
    seg = store.segments
    for s in steps.values():
        assert (np.diff(s.tau_s) > 0).all()
        np.testing.assert_array_equal(s.y_s[s.point], seg["y_s"][s.rows][s.point])
        np.testing.assert_array_equal(s.window, seg["window"][s.rows].astype(bool))
        assert abs(s.weight.sum() - 1.0) < 1e-9 and (s.weight[~s.point] == 0).all()
        inside, outside = s.weight[s.point & s.window], s.weight[s.point & ~s.window]
        if len(outside):
            assert np.allclose(outside / inside[0], data.FAR_WEIGHT)


def test_each_step_sees_the_latest_photo_already_available(toy_b6):
    _, store, _, _, steps, _ = toy_b6
    available = store.images["available_s"].astype(float)
    for unit, s in steps.items():
        mine = store.image_rows(unit=unit, kind=KINDS[0])
        for step in range(len(s.rows)):
            row = s.image_rows[step, 0]
            visible = mine[available[mine] < s.tau_s[step] - TIME_TOLERANCE_S]
            if row < 0:
                assert len(visible) == 0
            else:
                assert available[row] < s.tau_s[step] - TIME_TOLERANCE_S
                assert available[row] == available[visible].max()


def test_windows_end_at_their_point_and_never_look_ahead(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    units = store.unit_names[:3]
    windows = _windows(steps, features, units)
    assert len(windows) == sum(int(steps[u].point.sum()) for u in units)
    for item in (0, 5, len(windows) // 2, len(windows) - 1):
        sample = windows[item]
        u, p = windows.index[item]
        matrix = windows.matrices[u]
        start = p - data.INPUT_STEPS + 1
        lo = max(start, 0)
        np.testing.assert_array_equal(sample["force"][lo - start :], matrix[lo : p + 1])
        assert (sample["force"][: lo - start] == 0).all()  # left padding
        assert sample["row"] == steps[units[u]].rows[p]
        assert sample["images"].shape[1] == data.IMAGE_FEATURE_WIDTH
        assert (sample["images"][0] == 0).all()  # slot 0: no photo yet
        for step in range(data.INPUT_STEPS):
            unit_step = start + step
            slot = sample["image_indices"][step]
            if unit_step < 0 or steps[units[u]].image_rows[unit_step, 0] < 0:
                assert slot == 0
            else:
                row = steps[units[u]].image_rows[unit_step, 0]
                np.testing.assert_array_equal(sample["images"][slot], features[row])


def _batch(windows, n=4):
    return data.collate([windows[i] for i in range(n)])


def test_switched_off_inputs_cannot_change_the_output(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    windows = _windows(steps, features, store.unit_names[:2])
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    batch = _batch(windows)
    force, images, idx = batch["force"], batch["images"], batch["image_indices"]
    other_force = force.clone()
    other_force[..., : layout.descriptors] += 3.0
    other_usage = force.clone()
    other_usage[..., layout.usage] += 3.0
    cases = [
        (dict(use_force=False), (other_force, images, idx)),
        (dict(use_photos=False), (force, images + 5.0, idx)),
        (dict(use_usage=False), (other_usage, images, idx)),
    ]
    for switch, changed in cases:
        torch.manual_seed(0)
        model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, **switch).eval()
        with torch.no_grad():
            assert torch.allclose(model(force, images, idx), model(*changed), atol=1e-6), switch
    torch.manual_seed(0)
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH).eval()
    with torch.no_grad():
        for changed in ((other_force, images, idx), (force, images + 5.0, idx)):
            assert not torch.allclose(model(force, images, idx), model(*changed))


def test_the_model_keeps_omnifusers_encoder_and_replaces_its_heads(toy_b6):
    _, store, _, _, steps, _ = toy_b6
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    model = OmniFuserRUL(layout, 2, data.IMAGE_FEATURE_WIDTH)
    names = [name for name, _ in model.named_parameters()]
    assert not any("direct_force" in n or "direct_class" in n for n in names)
    assert not any(".encoder.features." in n for n in names)  # precomputed photo features
    assert model.backbone.image_branch.encoder.projection.in_features == 2 * 1280
    assert model.backbone.config.metadata == "none" and model.backbone.config.horizon == 1
    assert model.backbone.config.force_channels == layout.width


def test_select_epoch_uses_three_consecutive_values():
    assert select_epoch([5, 1, 9, 3, 3, 3]) == 5  # the dip at epoch 2 is a single value
    assert select_epoch([1, 2, 3]) == 2  # the centre of the only triple
    assert select_epoch([4, 2]) == 2


@pytest.fixture(scope="module")
def toy_fits(toy_b6):
    _, store, _, features, steps, root = toy_b6
    spec = ArmSpec(players=("force", "surface"), epochs=8, seeds=(0,))
    folder = run_arm(
        store,
        steps,
        features,
        KINDS,
        spec,
        root / "runs",
        folds=[1],
        device="cpu",
        log=lambda *_: None,
    )
    return spec, folder


def test_a_fit_learns_the_toy_rul_and_a_shuffled_control_does_not(toy_b6, toy_fits, tmp_path):
    _, store, _, features, steps, root = toy_b6
    spec, folder = toy_fits
    fold = leave_one_unit_out(store.unit_names)[0]
    validation = steps[fold.validation]
    train_y = np.concatenate([steps[u].y_s[steps[u].point] for u in fold.train])
    v = validation.point
    constant = float(
        np.average(np.abs(train_y.mean() - validation.y_s[v]), weights=validation.weight[v])
    )
    import json

    learned = min(
        json.loads((folder / "fold-01" / "seed-0" / "fit.json").read_text())["validation_mae_s"]
    )
    assert learned < 0.7 * constant
    rng = np.random.default_rng(0)
    shuffled = dict(steps)
    for u in fold.train:
        s = steps[u]
        y = s.y_s.copy()
        y[s.point] = rng.permutation(
            np.concatenate([steps[t].y_s[steps[t].point] for t in fold.train])
        )[: int(s.point.sum())]
        shuffled[u] = replace(s, y_s=y)
    control = run_arm(
        store,
        shuffled,
        features,
        KINDS,
        spec,
        tmp_path / "control",
        folds=[1],
        device="cpu",
        log=lambda *_: None,
    )
    control_mae = min(
        json.loads((control / "fold-01" / "seed-0" / "fit.json").read_text())["validation_mae_s"]
    )
    assert control_mae > 0.8 * constant


def test_predictions_cover_validation_and_test_and_runs_resume(toy_b6, toy_fits):
    _, store, _, features, steps, root = toy_b6
    spec, folder = toy_fits
    fold = leave_one_unit_out(store.unit_names)[0]
    table = pq.read_table(folder / "fold-01" / "seed-0" / "predictions.parquet").to_pydict()
    roles = np.asarray(table["role"])
    assert (roles == "validation").sum() == steps[fold.validation].point.sum()
    assert (roles == "test").sum() == steps[fold.test].point.sum()
    assert set(np.asarray(table["unit"])[roles == "test"]) == {fold.test}
    prediction = np.asarray(table["prediction_s"])
    assert (prediction >= 0).all() and (prediction <= store.settings.horizon_s).all()
    messages = []
    run_arm(
        store,
        steps,
        features,
        KINDS,
        spec,
        root / "runs",
        folds=[1],
        device="cpu",
        log=messages.append,
    )
    assert any("done, skipped" in m for m in messages)


def test_the_same_seed_gives_the_same_predictions(toy_b6, tmp_path):
    _, store, _, features, steps, _ = toy_b6
    spec = ArmSpec(players=("force",), epochs=2, seeds=(3,))
    a = run_arm(
        store,
        steps,
        features,
        KINDS,
        spec,
        tmp_path / "a",
        folds=[2],
        device="cpu",
        log=lambda *_: None,
    )
    b = run_arm(
        store,
        steps,
        features,
        KINDS,
        spec,
        tmp_path / "b",
        folds=[2],
        device="cpu",
        log=lambda *_: None,
    )
    pa = (
        pq.read_table(a / "fold-02" / "seed-3" / "predictions.parquet")
        .column("prediction_s")
        .to_numpy()
    )
    pb = (
        pq.read_table(b / "fold-02" / "seed-3" / "predictions.parquet")
        .column("prediction_s")
        .to_numpy()
    )
    np.testing.assert_array_equal(pa, pb)


def test_test_scores_are_sealed_until_the_freeze(toy_b6, toy_fits):
    _, store, *_ = toy_b6
    _, folder = toy_fits
    validation = aggregate.score_arm(store, folder, "validation")
    assert validation["summary"]["window_mae"]["mean"] > 0
    with pytest.raises(aggregate.TestSealed):
        aggregate.score_arm(store, folder, "test")
    with pytest.raises(aggregate.TestSealed):
        aggregate.score_arm(store, folder, "test", after_freeze=True, today=date(2026, 11, 30))
    after = aggregate.score_arm(store, folder, "test", after_freeze=True, today=date(2026, 12, 1))
    assert set(after["per_unit"]) == {leave_one_unit_out(store.unit_names)[0].test}


def test_the_ensemble_averages_the_seeds():
    predictions = {
        "role": np.array(["test"] * 4),
        "fold": np.array([1, 1, 1, 1]),
        "row": np.array([7, 8, 7, 8]),
        "unit": np.array(["U1"] * 4),
        "prediction_s": np.array([100.0, 300.0, 200.0, 500.0]),
    }
    combined = aggregate.ensemble(predictions, "test")
    np.testing.assert_array_equal(combined["row"], [7, 8])
    np.testing.assert_allclose(combined["prediction_s"], [150.0, 400.0])
    assert (combined["seeds"] == 2).all()


def test_shapley_values_are_exact():
    contributions = {"a": 3.0, "b": 5.0, "c": -1.0}
    phi = aggregate.shapley(lambda s: sum(contributions[p] for p in s), list(contributions))
    assert phi == pytest.approx(contributions)
    pair_only = aggregate.shapley(lambda s: 6.0 if {"a", "b"} <= s else 0.0, ["a", "b", "c"])
    assert pair_only == pytest.approx({"a": 3.0, "b": 3.0, "c": 0.0})


def test_two_modalities_are_judged_on_a_preset_paired_test():
    units = [f"U{i}" for i in range(10)]
    single, pair, full = (
        frozenset(["force"]),
        frozenset(["force", "tool"]),
        frozenset(["force", "tool", "chip"]),
    )
    window = {
        frozenset(["force"]): {u: 400.0 for u in units},
        frozenset(["tool"]): {u: 380.0 for u in units},
        frozenset(["chip"]): {u: 450.0 for u in units},
        pair: {u: 340.0 + i for i, u in enumerate(units)},
        frozenset(["force", "chip"]): {u: 390.0 for u in units},
        frozenset(["tool", "chip"]): {u: 370.0 for u in units},
        full: {u: 338.0 + i for i, u in enumerate(units)},
    }
    validation = {s: float(np.mean(list(v.values()))) for s, v in window.items()}
    report = aggregate.modality_report(
        window, validation, ["force", "tool", "chip"], {u: 470.0 for u in units}
    )
    assert report["best_single"] == ["tool"] and report["best_pair"] == ["force", "tool"]
    assert report["pair_vs_single"]["justified"]  # ≥ 10 s better on every unit
    assert not report["all_vs_pair"]["justified"]  # 2 s better: below the threshold
    total = sum(report["shapley"][p]["mean"] for p in ["force", "tool", "chip"])
    assert total == pytest.approx(470.0 - np.mean(list(window[full].values())))  # efficiency
    assert single in window


def test_the_study_arms():
    names = [spec.name for spec in modality_arms(["force", "surface"], ArmSpec(players=()))]
    assert names == [
        "usage-only",
        "force",
        "surface",
        "force+surface",
        "force-nousage",
        "surface-nousage",
        "force+surface-dropout0.2",
    ]


def test_samples_cover_the_baselines_points(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    seg = store.segments
    for unit in store.unit_names:
        windows = _windows(steps, features, [unit])
        rows = np.array([steps[unit].rows[p] for _, p in windows.index])
        expected = np.flatnonzero((seg["unit"] == unit) & seg["is_prediction_point"].astype(bool))
        np.testing.assert_array_equal(np.sort(rows), expected)
    usage = score_usage_baselines(store, leave_one_unit_out(store.unit_names))
    b2_test = aggregate.b2_by_unit(store, "test")
    for row in usage["rows"]:
        if row["baseline"] == "B2":
            assert row["mae"] == pytest.approx(b2_test[row["fold_test"]])


def test_arm_folders_are_per_store_and_arm(toy_b6, tmp_path):
    _, store, *_ = toy_b6
    spec = ArmSpec(players=("force",), usage=False)
    assert arm_dir(tmp_path, store, spec) == tmp_path / f"toy-{store.signature}" / "force-nousage"


@pytest.mark.data
def test_photos_of_one_recording_become_visible_together():
    from rulfuse.adapters import load_dataset

    store = SegmentStore.open(build_segment_store(load_dataset("nonastreda")))
    images = store.images
    by_recording: dict[tuple[str, str], set[float]] = {}
    for unit, key, available in zip(
        images["unit"], images["key"], images["available_s"], strict=True
    ):
        by_recording.setdefault((str(unit), str(key)), set()).add(round(float(available), 6))
    assert by_recording and all(len(times) == 1 for times in by_recording.values())


def test_a_cache_key_with_tuples_is_found_again(tmp_path):
    key = {"specs": {"tool": {"crop": (0.3, 0.0, 0.85, 1.0)}}, "version": 1}
    data._save(tmp_path, np.arange(3.0), {"key": key})
    np.testing.assert_array_equal(data._load(tmp_path, key), np.arange(3.0))
    with pytest.raises(ValueError, match="mismatch"):
        data._load(tmp_path, {**key, "version": 2})
