"""OmniFuser-RUL, baseline B6 (docs/OMNIFUSER_RUL_PLAN.md §9; version 3): the checks on the
toy dataset (at 10 s, and at 2 s against its 10 s grid) and on the real store."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date

import numpy as np
import pyarrow.parquet as pq
import pytest

# B6 needs the `omnifuser` extra (OmniFuser v1.1 and torchvision). Without it these tests
# skip instead of breaking the collection of the whole suite (Ruche job J0 runs it).
pytest.importorskip("torchvision", reason="B6 needs torchvision (the omnifuser extra)")
pytest.importorskip("omnifuser", reason="B6 needs OmniFuser v1.1 (the omnifuser extra)")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from rulfuse.adapters import toy  # noqa: E402
from rulfuse.baselines import score_usage_baselines  # noqa: E402
from rulfuse.core.folds import leave_one_unit_out  # noqa: E402
from rulfuse.core.schema import TIME_TOLERANCE_S  # noqa: E402
from rulfuse.core.store import SegmentStore, build_segment_store, grid_mask  # noqa: E402
from rulfuse.features.cache import extract_signal_features  # noqa: E402
from rulfuse.features.encoders import ProjectionEncoder  # noqa: E402
from rulfuse.omnifuser_rul import aggregate, data  # noqa: E402
from rulfuse.omnifuser_rul.model import DROP_FORCE, MemorySpec, OmniFuserRUL  # noqa: E402
from rulfuse.omnifuser_rul.study import modality_arms  # noqa: E402
from rulfuse.omnifuser_rul.train import (  # noqa: E402
    ArmSpec,
    arm_dir,
    predict_unit,
    run_arm,
    select_epoch,
)

KINDS = ["surface"]
QUICK = {"pairs_per_epoch": 128}  # small epochs for CPU tests


def quiet(*_: object) -> None:
    return None


class FakePhotoEncoder(nn.Module):
    """1,280 features that follow image brightness (no download)."""

    def backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return mean * torch.linspace(0.5, 1.5, data.IMAGE_FEATURE_WIDTH)[None]


def _toy(root, segment_s=10.0):
    ds = toy.build(root / "images", n_units=5, segment_s=segment_s)
    store = SegmentStore.open(build_segment_store(ds, root / f"store-{segment_s:g}"))
    descriptors = data.descriptor_cache(store, ds.signals, log=quiet, root=root / "f")
    features = data.image_feature_cache(store, encoder=FakePhotoEncoder(), log=quiet, root=root / "f")
    steps = data.build_steps(store, descriptors, KINDS)
    return ds, store, descriptors, features, steps


@pytest.fixture(scope="module")
def toy_b6(tmp_path_factory):
    root = tmp_path_factory.mktemp("b6")
    return (*_toy(root), root)


@pytest.fixture(scope="module")
def toy_fine(tmp_path_factory):
    """The same toy units cut at 2 s; its evaluation grid is the 10 s store's points."""
    root = tmp_path_factory.mktemp("b6fine")
    return (*_toy(root, 2.0), root)


def _inputs(steps, features, units, kinds_used=(0,)):
    standardizer = data.Standardizer.fit([steps[u] for u in units])
    return [data.unit_inputs(steps[u], standardizer, features, list(kinds_used)) for u in units]


# ----------------------------------------------------------------------------- data


def test_descriptors_are_the_feature_caches_descriptors(toy_b6, tmp_path):
    ds, store, descriptors, *_, root = toy_b6
    signal = extract_signal_features(store, ds.signals, ProjectionEncoder(), root=tmp_path, log=quiet)
    np.testing.assert_allclose(descriptors, np.asarray(signal.descriptors), rtol=1e-6)
    again = data.descriptor_cache(store, ds.signals, log=quiet, root=root / "f")
    assert again is not descriptors  # read back from the cache
    np.testing.assert_array_equal(again, descriptors)


def test_steps_follow_the_store_targets_and_weights(toy_b6):
    _, store, _, _, steps, _ = toy_b6
    seg = store.segments
    for s in steps.values():
        assert (np.diff(s.tau_s) > 0).all()
        assert s.point[: s.n_points].all()  # the points are a prefix of the steps
        np.testing.assert_array_equal(s.y_s[s.point], seg["y_s"][s.rows][s.point])
        np.testing.assert_array_equal(s.window, seg["window"][s.rows].astype(bool))
        assert abs(s.weight.sum() - 1.0) < 1e-9 and (s.weight[~s.point] == 0).all()
        inside, outside = s.weight[s.point & s.window], s.weight[s.point & ~s.window]
        if len(outside):
            assert np.allclose(outside / inside[0], data.FAR_WEIGHT)
        np.testing.assert_array_equal(s.grid, s.point)  # a 10 s store is its own grid


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
    inputs = _inputs(steps, features, units)
    tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    for i, u in enumerate(inputs):
        assert u.n == steps[units[i]].n_points
        assert (u.composites[0] == 0).all()  # row 0: no photo yet
        for p in (0, 5, u.n // 2, u.n - 1):
            force, images, idx = tensors.windows(torch.tensor([i]), torch.tensor([p]))
            force, images, idx = force[0].numpy(), images[0].numpy(), idx[0].numpy()
            start = p - data.INPUT_STEPS + 1
            lo = max(start, 0)
            np.testing.assert_array_equal(force[lo - start :], u.force[lo : p + 1])
            assert (force[: lo - start] == 0).all() and (idx[: lo - start] == 0).all()
            for step in range(lo, p + 1):
                photo = steps[units[i]].image_rows[step, 0]
                slot = idx[step - start]
                if photo < 0:
                    assert slot == 0
                else:
                    np.testing.assert_array_equal(images[slot], features[photo])


def test_pairs_are_ordered_and_within_one_unit(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:3])
    pairs = data.sample_pairs(inputs, 4000, 50, np.random.default_rng(0))
    unit, earlier, later = pairs.T
    sizes = np.array([u.n for u in inputs])
    assert ((0 <= earlier) & (earlier <= later) & (later < sizes[unit])).all()
    lag = later - earlier
    assert 0.4 < np.mean(lag == 1) < 0.6 and lag.max() <= 50  # the previous segment, half


# ----------------------------------------------------------------------------- grid


def test_a_fine_store_contains_the_10s_grid_with_the_same_labels(toy_b6, toy_fine):
    _, coarse, *_ = toy_b6
    _, fine, _, _, fine_steps, _ = toy_fine
    on = grid_mask(fine, 10.0)
    assert len(fine.segments["unit"]) > 4 * len(coarse.segments["unit"])
    c, f = coarse.segments, fine.segments
    key = lambda seg, rows: {  # noqa: E731
        (str(seg["unit"][r]), str(seg["recording_key"][r]), round(float(seg["elapsed_s"][r]), 6)): r
        for r in rows
    }
    fine_keys, coarse_keys = key(f, np.flatnonzero(on)), key(c, range(len(c["unit"])))
    assert fine_keys.keys() == coarse_keys.keys()  # exactly the 10 s points
    for k, rc in coarse_keys.items():
        rf = fine_keys[k]
        assert f["tau_s"][rf] == c["tau_s"][rc]
        if c["is_prediction_point"][rc]:
            assert f["rul_s"][rf] == pytest.approx(c["rul_s"][rc], abs=1e-9)
            assert f["window"][rf] == c["window"][rc]
    assert all(fine.units[u]["failure_s"] == coarse.units[u]["failure_s"] for u in coarse.units)
    # the steps' grid flags and grid weights are the 10 s store's points and weights
    for unit, s in fine_steps.items():
        assert s.grid.sum() == int(c["is_prediction_point"][c["unit"] == unit].sum())
        assert abs(s.grid_weight.sum() - 1.0) < 1e-9


def test_b2_on_the_grid_equals_b2_on_the_coarse_store(toy_b6, toy_fine):
    _, coarse, *_ = toy_b6
    _, fine, *_ = toy_fine
    for role in ("validation", "test"):
        assert aggregate.b2_by_unit(fine, role) == pytest.approx(aggregate.b2_by_unit(coarse, role))


# ----------------------------------------------------------------------------- model


def test_omnifuser_input_dimensions_do_not_depend_on_the_segment_length(toy_b6, toy_fine):
    counts = []
    for fixture in (toy_b6, toy_fine):
        _, store, _, features, steps, _ = fixture
        inputs = _inputs(steps, features, store.unit_names[:2])
        tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
        force, images, idx = tensors.windows(torch.tensor([0, 1]), torch.tensor([3, 40]))
        layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
        # toy: 2 channels x 7 descriptors + signal flag + 2 usage = 17 (Nonastreda: 4 x 7 + 3 = 31)
        assert force.shape == (2, data.INPUT_STEPS, layout.width) == (2, 96, 2 * 7 + 1 + 2)
        assert idx.shape == (2, 96) and images.shape[-1] == data.IMAGE_FEATURE_WIDTH
        model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec()).eval()
        with torch.no_grad():
            memory = model.memory_states(*tensors.memory_inputs())
            out = model(force, images, idx, memory[torch.tensor([0, 1]), torch.tensor([3, 40])])
        assert out.shape == (2,) and torch.isfinite(out).all()
        counts.append(sum(p.numel() for p in model.backbone.parameters()))
    assert counts[0] == counts[1]  # OmniFuser itself is the same at 10 s and 2 s steps


def test_switched_off_inputs_cannot_change_the_output_or_the_memory(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:2])
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])

    def run(model, tensors):
        unit, step = torch.tensor([0, 0, 1, 1]), torch.tensor([10, 30, 20, 50])
        force, images, idx = tensors.windows(unit, step)
        memory = model.memory_states(*tensors.memory_inputs())
        return model(force, images, idx, memory[unit, step])

    def changed(what):
        out = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
        if what == "force":
            out.force[..., : layout.descriptors] += 3.0
        elif what == "usage":
            out.force[..., layout.usage] += 3.0
        else:
            out.composites += 5.0
        return out

    base = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    cases = {"force": dict(use_force=False), "photos": dict(use_photos=False), "usage": dict(use_usage=False)}
    for what, switch in cases.items():
        torch.manual_seed(0)
        model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec(), **switch).eval()
        with torch.no_grad():
            assert torch.allclose(run(model, base), run(model, changed(what)), atol=1e-5), what
    torch.manual_seed(0)
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec()).eval()
    with torch.no_grad():
        for what in cases:
            assert not torch.allclose(run(model, base), run(model, changed(what))), what


def test_the_model_keeps_omnifusers_encoder_and_adds_the_memory(toy_b6):
    _, store, _, _, steps, _ = toy_b6
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    model = OmniFuserRUL(layout, 2, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec(hidden=48))
    names = [name for name, _ in model.named_parameters()]
    assert not any("direct_force" in n or "direct_class" in n for n in names)
    assert not any(".encoder.features." in n for n in names)  # precomputed photo features
    assert model.backbone.image_branch.encoder.projection.in_features == 2 * 1280
    assert model.backbone.config.metadata == "none" and model.backbone.config.horizon == 1
    assert model.backbone.config.force_channels == layout.width
    assert model.head.in_features == model.backbone.config.d_model + 48
    plain = OmniFuserRUL(layout, 2, data.IMAGE_FEATURE_WIDTH)
    assert plain.head.in_features == plain.backbone.config.d_model and not hasattr(plain, "memory")


def test_modality_dropout_reads_the_matching_memory(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:1])
    tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec()).eval()
    with torch.no_grad():
        full = model.memory_states(*tensors.memory_inputs())
        no_force = model.memory_states(*tensors.memory_inputs(), variant=DROP_FORCE)
        off = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec(), use_force=False)
        off.load_state_dict(model.state_dict())
        assert torch.allclose(no_force, off.eval().memory_states(*tensors.memory_inputs()))
        assert not torch.allclose(full, no_force)


def test_select_epoch_uses_three_consecutive_values():
    assert select_epoch([5, 1, 9, 3, 3, 3]) == 5  # the dip at epoch 2 is a single value
    assert select_epoch([1, 2, 3]) == 2  # the centre of the only triple
    assert select_epoch([4, 2]) == 2


# ----------------------------------------------------------------------------- fits


@pytest.fixture(scope="module")
def toy_fits(toy_b6):
    _, store, _, features, steps, root = toy_b6
    spec = ArmSpec(players=("force", "surface"), epochs=8, seeds=(0,), **QUICK)
    folder = run_arm(store, steps, features, KINDS, spec, root / "runs", folds=[1], device="cpu", log=quiet)
    return spec, folder


def _best(folder, fold="fold-01", seed="seed-0"):
    return min(json.loads((folder / fold / seed / "fit.json").read_text())["validation_mae_s"])


def test_a_fit_learns_the_toy_rul_and_a_shuffled_control_does_not(toy_b6, toy_fits, tmp_path):
    _, store, _, features, steps, root = toy_b6
    spec, folder = toy_fits
    fold = leave_one_unit_out(store.unit_names)[0]
    validation = steps[fold.validation]
    train_y = np.concatenate([steps[u].y_s[steps[u].point] for u in fold.train])
    v = validation.point
    constant = float(np.average(np.abs(train_y.mean() - validation.y_s[v]), weights=validation.weight[v]))
    assert _best(folder) < 0.7 * constant
    rng = np.random.default_rng(0)
    shuffled = dict(steps)
    for u in fold.train:
        s = steps[u]
        y = s.y_s.copy()
        y[s.point] = rng.permutation(
            np.concatenate([steps[t].y_s[steps[t].point] for t in fold.train])
        )[: int(s.point.sum())]
        shuffled[u] = replace(s, y_s=y)
    control = run_arm(store, shuffled, features, KINDS, spec, tmp_path / "control", folds=[1], device="cpu", log=quiet)
    assert _best(control) > 0.8 * constant


def test_predictions_cover_every_point_with_the_grid_flag_and_runs_resume(toy_b6, toy_fits):
    _, store, _, features, steps, root = toy_b6
    spec, folder = toy_fits
    fold = leave_one_unit_out(store.unit_names)[0]
    table = pq.read_table(folder / "fold-01" / "seed-0" / "predictions.parquet").to_pydict()
    roles = np.asarray(table["role"])
    assert (roles == "validation").sum() == steps[fold.validation].point.sum()
    assert (roles == "test").sum() == steps[fold.test].point.sum()
    assert set(np.asarray(table["unit"])[roles == "test"]) == {fold.test}
    assert np.asarray(table["on_grid"]).all()  # a 10 s store: every point is a grid point
    prediction = np.asarray(table["prediction_s"])
    assert (prediction >= 0).all() and (prediction <= store.settings.horizon_s).all()
    record = json.loads((folder / "fold-01" / "seed-0" / "fit.json").read_text())
    assert len(record["train_loss"]) == spec.epochs and {"mse", "hinge", "slope"} <= set(record["train_loss"][0])
    messages = []
    run_arm(store, steps, features, KINDS, spec, root / "runs", folds=[1], device="cpu", log=messages.append)
    assert any("done, skipped" in m for m in messages)


def test_a_fine_fit_predicts_every_1s_point_and_selects_on_the_grid(toy_fine):
    _, store, _, features, steps, root = toy_fine
    spec = ArmSpec(players=("force", "surface"), epochs=3, seeds=(0,), **QUICK)
    folder = run_arm(store, steps, features, KINDS, spec, root / "runs", folds=[1], device="cpu", log=quiet)
    table = pq.read_table(folder / "fold-01" / "seed-0" / "predictions.parquet").to_pydict()
    fold = leave_one_unit_out(store.unit_names)[0]
    roles, grid = np.asarray(table["role"]), np.asarray(table["on_grid"])
    validation = steps[fold.validation]
    assert (roles == "validation").sum() == validation.n_points
    assert grid[roles == "validation"].sum() == validation.grid.sum() < validation.n_points
    scores = aggregate.score_arm(store, folder, "validation")
    unit = scores["per_unit"][fold.validation]
    assert unit["points"] == validation.grid.sum() and unit["points_all"] == validation.n_points
    assert {"increase_rate", "mean_increase_s", "slope_error_s"} <= set(unit)


def test_the_same_seed_gives_the_same_predictions(toy_b6, tmp_path):
    _, store, _, features, steps, _ = toy_b6
    spec = ArmSpec(players=("force",), epochs=2, seeds=(3,), **QUICK)
    a = run_arm(store, steps, features, KINDS, spec, tmp_path / "a", folds=[2], device="cpu", log=quiet)
    b = run_arm(store, steps, features, KINDS, spec, tmp_path / "b", folds=[2], device="cpu", log=quiet)
    read = lambda f: pq.read_table(f / "fold-02" / "seed-3" / "predictions.parquet").column("prediction_s").to_numpy()  # noqa: E731
    np.testing.assert_array_equal(read(a), read(b))


def test_the_temporal_terms_reduce_rising_predictions(toy_b6, tmp_path):
    """The monotonic hinge and the slope term, against the same fit without them (2 seeds)."""
    _, store, _, features, steps, _ = toy_b6
    rates = {}
    for name, weights in (("on", {}), ("off", {"mono_weight": 0.0, "slope_weight": 0.0})):
        spec = ArmSpec(players=("force", "surface"), epochs=5, seeds=(0, 1), **QUICK, **weights)
        folder = run_arm(store, steps, features, KINDS, spec, tmp_path / name, folds=[1], device="cpu", log=quiet)
        unit = aggregate.score_arm(store, folder, "validation")["per_unit"]
        rates[name] = np.mean([u["mean_increase_s"] for u in unit.values()])
    assert rates["on"] < rates["off"]


def test_predict_unit_matches_the_training_memory(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:2])
    tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec()).eval()
    one = data.FitTensors(inputs[1:], data.INPUT_STEPS, "cpu")
    points = np.arange(inputs[1].n)
    # the memory of a unit does not depend on which other units share the tensors
    np.testing.assert_allclose(
        predict_unit(model, tensors, 1, points, 16), predict_unit(model, one, 0, points, 16), atol=1e-5
    )


# ----------------------------------------------------------------------------- scores


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
        "on_grid": np.array([True, False, True, False]),
        "prediction_s": np.array([100.0, 300.0, 200.0, 500.0]),
    }
    combined = aggregate.ensemble(predictions, "test")
    np.testing.assert_array_equal(combined["row"], [7, 8])
    np.testing.assert_allclose(combined["prediction_s"], [150.0, 400.0])
    np.testing.assert_array_equal(combined["on_grid"], [True, False])
    assert (combined["seeds"] == 2).all()


def test_shapley_values_are_exact():
    contributions = {"a": 3.0, "b": 5.0, "c": -1.0}
    phi = aggregate.shapley(lambda s: sum(contributions[p] for p in s), list(contributions))
    assert phi == pytest.approx(contributions)
    pair_only = aggregate.shapley(lambda s: 6.0 if {"a", "b"} <= s else 0.0, ["a", "b", "c"])
    assert pair_only == pytest.approx({"a": 3.0, "b": 3.0, "c": 0.0})


def test_two_modalities_are_judged_on_a_preset_paired_test():
    units = [f"U{i}" for i in range(10)]
    pair, full = frozenset(["force", "tool"]), frozenset(["force", "tool", "chip"])
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
    report = aggregate.modality_report(window, validation, ["force", "tool", "chip"], {u: 470.0 for u in units})
    assert report["best_single"] == ["tool"] and report["best_pair"] == ["force", "tool"]
    assert report["pair_vs_single"]["justified"]  # ≥ 10 s better on every unit
    assert not report["all_vs_pair"]["justified"]  # 2 s better: below the threshold
    total = sum(report["shapley"][p]["mean"] for p in ["force", "tool", "chip"])
    assert total == pytest.approx(470.0 - np.mean(list(window[full].values())))  # efficiency


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
        "force+surface-nomemory",
        "force+surface-notemporal",
    ]


def test_samples_cover_the_baselines_points(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    seg = store.segments
    for unit in store.unit_names:
        (inputs,) = _inputs(steps, features, [unit])
        rows = steps[unit].rows[: inputs.n]
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
    assert ArmSpec(players=("force",), memory=False).name == "force-nomemory"
    assert ArmSpec(players=("force",), mono_weight=0, slope_weight=0).name == "force-notemporal"


def test_a_cache_key_with_tuples_is_found_again(tmp_path):
    key = {"specs": {"tool": {"crop": (0.3, 0.0, 0.85, 1.0)}}, "version": 1}
    data._save(tmp_path, np.arange(3.0), {"key": key})
    np.testing.assert_array_equal(data._load(tmp_path, key), np.arange(3.0))
    with pytest.raises(ValueError, match="mismatch"):
        data._load(tmp_path, {**key, "version": 2})


def test_photo_features_are_shared_by_stores_of_any_segment_length(toy_b6, toy_fine):
    assert data.image_table_key(toy_b6[1]) == data.image_table_key(toy_fine[1])


# ----------------------------------------------------------------------------- real data


@pytest.mark.data
def test_photos_of_one_recording_become_visible_together():
    from rulfuse.adapters import load_dataset

    store = SegmentStore.open(build_segment_store(load_dataset("nonastreda")))
    images = store.images
    by_recording: dict[tuple[str, str], set[float]] = {}
    for unit, key, available in zip(images["unit"], images["key"], images["available_s"], strict=True):
        by_recording.setdefault((str(unit), str(key)), set()).add(round(float(available), 6))
    assert by_recording and all(len(times) == 1 for times in by_recording.values())


@pytest.mark.data
def test_the_1s_store_scores_b2_exactly_as_the_plan(tmp_path):
    """B6's 1 s store, scored on its 10 s grid, gives B2 the plan's numbers (§2.7)."""
    from rulfuse.adapters.nonastreda import load_data, to_dataset

    data_view = load_data()
    fine = SegmentStore.open(build_segment_store(to_dataset(data_view, 1.0), tmp_path / "s1"))
    coarse = SegmentStore.open(build_segment_store(to_dataset(data_view, 10.0), tmp_path / "s10"))
    assert int(fine.segments["is_prediction_point"].sum()) == 36919
    assert int(grid_mask(fine, 10.0).sum()) == len(coarse.segments["unit"]) == 5125
    usage = score_usage_baselines(coarse, leave_one_unit_out(coarse.unit_names))
    b2_test = aggregate.b2_by_unit(fine, "test")
    assert np.mean(list(b2_test.values())) == pytest.approx(usage["summary"]["B2"]["mae"])
    assert usage["summary"]["B2"]["mae"] == pytest.approx(467.8, abs=0.5)


# ----------------------------------------------------------------------------- results


def _fake_study(store, root, players, bias, seeds=(0,)):
    """Write every fit of a study as ``fit()`` does, with predictions = target + the arm's
    bias, so each arm's window MAE is known exactly."""
    import pyarrow as pa

    base = ArmSpec(players=(), seeds=seeds, epochs=5)
    seg = store.segments
    folds = leave_one_unit_out(store.unit_names)
    for spec in modality_arms(players, base):
        for index, fold in enumerate(folds, start=1):
            for seed in seeds:
                out = arm_dir(root, store, spec) / f"fold-{index:02d}" / f"seed-{seed}"
                out.mkdir(parents=True, exist_ok=True)
                rows, units, roles = [], [], []
                for role, unit in (("validation", fold.validation), ("test", fold.test)):
                    mine = np.flatnonzero((seg["unit"] == unit) & seg["is_prediction_point"])
                    rows.append(mine)
                    units += [unit] * len(mine)
                    roles += [role] * len(mine)
                rows_all = np.concatenate(rows)
                table = pa.table(
                    {
                        "row": rows_all,
                        "unit": units,
                        "role": roles,
                        "fold": np.full(len(roles), index),
                        "seed": np.full(len(roles), seed),
                        "on_grid": np.ones(len(roles), dtype=bool),
                        "prediction_s": seg["y_s"][rows_all] + bias[spec.name],
                    }
                )
                pq.write_table(table, out / "predictions.parquet")
                record = {
                    "fold": index,
                    "seed": seed,
                    "selected_epoch": 2,
                    "validation_mae_s": [9.0, 8.0, 7.5, 7.6, 7.7],
                    "seconds": 12.0,
                    "spec": spec.signature(),
                }
                (out / "fit.json").write_text(json.dumps(record), encoding="utf-8")
    return base


BIAS = {
    "usage-only": 400.0,
    "force": 120.0,
    "surface": 60.0,
    "force+surface": 30.0,
    "force-nousage": 200.0,
    "surface-nousage": 150.0,
    "force+surface-dropout0.2": 35.0,
    "force+surface-nomemory": 45.0,
    "force+surface-notemporal": 40.0,
}


def test_results_report_progress_scores_and_choices_before_the_freeze(toy_b6, tmp_path):
    from rulfuse.omnifuser_rul.report import write_results

    _, store, *_ = toy_b6
    players = ["force", "surface"]
    base = _fake_study(store, tmp_path, players, BIAS)
    out = write_results(store, tmp_path, players, base, today=date(2026, 11, 20))
    progress = {a["arm"]: a for a in json.loads((out / "progress.json").read_text())}
    assert set(progress) == set(BIAS) and all(a["complete"] for a in progress.values())
    assert all(a["done"] == a["expected"] == 5 for a in progress.values())  # 5 folds x 1 seed
    validation = json.loads((out / "analysis-validation.json").read_text())["validation"]
    for arm, bias in BIAS.items():  # predictions = y + bias: the window MAE is the bias
        assert validation[arm]["summary"]["window_mae"]["mean"] == pytest.approx(bias)
    choices = json.loads((out / "choices.json").read_text())
    assert choices["best_single"] == ["surface"] and choices["best_pair"] == ["force", "surface"]
    assert choices["complete"] is True
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "Sealed until 2026-12-01" in text and "| surface | 5/5 |" in text
    assert not (out / "analysis-test.json").exists()  # nothing about the test units yet
    curves = (out / "curves.csv").read_text().splitlines()
    assert len(curves) == 1 + len(BIAS) * 5 * 5  # header + arms x fits x epochs


def test_results_test_read_needs_the_freeze_and_a_complete_study(toy_b6, tmp_path):
    from rulfuse.omnifuser_rul.report import IncompleteStudy, write_results

    _, store, *_ = toy_b6
    players = ["force", "surface"]
    base = _fake_study(store, tmp_path, players, BIAS)
    write_results(store, tmp_path, players, base)  # records the choices before the test read
    with pytest.raises(aggregate.TestSealed):
        write_results(store, tmp_path, players, base, after_freeze=True, today=date(2026, 11, 30))
    out = write_results(store, tmp_path, players, base, after_freeze=True, today=date(2026, 12, 1))
    test = json.loads((out / "analysis-test.json").read_text())
    assert test["test"]["surface"]["summary"]["window_mae"]["mean"] == pytest.approx(60.0)
    shapley = test["modality"]["shapley"]
    # Two players, v(S) = -MAE: φ_surface - φ_force = v(surface) - v(force) = 120 - 60 = 60 s
    # whatever v(∅) is; efficiency: φ_surface + φ_force = MAE(∅) - MAE(both).
    assert shapley["surface"]["mean"] - shapley["force"]["mean"] == pytest.approx(60.0)
    assert test["empty_coalition"] == "B2"  # B2 beats the usage-only arm (bias 400 s)
    b2_test = np.mean(list(aggregate.b2_by_unit(store, "test").values()))
    total = shapley["surface"]["mean"] + shapley["force"]["mean"]
    assert total == pytest.approx(b2_test - 30.0)
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "Shapley values" in text and "recorded before the test read: **True**" in text
    missing = arm_dir(tmp_path, store, replace(base, players=("force",))) / "fold-03" / "seed-0"
    (missing / "fit.json").unlink()  # one fit lost: no test read on an incomplete study
    with pytest.raises(IncompleteStudy):
        write_results(store, tmp_path, players, base, after_freeze=True, today=date(2026, 12, 1))
