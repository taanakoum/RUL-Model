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
    pair_loss,
    predict_unit,
    run_arm,
    select_epoch,
)

KINDS = ["surface"]
QUICK_PAIRS = 128  # small epochs for CPU tests
SHORT_BLOCKS = MemorySpec(block=16)  # toy units have 82-109 points: several 16-step blocks


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
    features = data.image_feature_cache(
        store, encoder=FakePhotoEncoder(), log=quiet, root=root / "f"
    )
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
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=tmp_path, log=quiet
    )
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
    cases = {
        "force": dict(use_force=False),
        "photos": dict(use_photos=False),
        "usage": dict(use_usage=False),
    }
    for what, switch in cases.items():
        torch.manual_seed(0)
        model = OmniFuserRUL(
            layout, 1, data.IMAGE_FEATURE_WIDTH, memory=SHORT_BLOCKS, **switch
        ).eval()
        with torch.no_grad():
            assert torch.allclose(run(model, base), run(model, changed(what)), atol=1e-5), what
    torch.manual_seed(0)
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=SHORT_BLOCKS).eval()
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
    # the memory reads OmniFuser's two encoded streams; it has no input layer of its own
    assert model.memory.rnn.input_size == 2 * model.backbone.config.d_model
    assert model.memory.block == data.INPUT_STEPS and not hasattr(model, "photo_projection")
    plain = OmniFuserRUL(layout, 2, data.IMAGE_FEATURE_WIDTH)
    assert plain.head.in_features == plain.backbone.config.d_model and not hasattr(plain, "memory")


def _memory_case(toy_b6, block=16):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:2])
    tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    torch.manual_seed(0)
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=MemorySpec(block=block))
    return inputs, tensors, model.eval()


def test_the_memory_reads_omnifusers_encodings_of_each_block(toy_b6):
    """The GRU's input for block j is the mean, over the head's own window ending at the
    block's last step, of the force and photo streams that OmniFuser.encode fuses."""
    inputs, tensors, model = _memory_case(toy_b6)
    backbone = model.backbone
    unit, step = torch.tensor([0, 1]), torch.tensor([47, 63])
    force, images, idx = tensors.windows(unit, step)
    captured = {}
    hooks = [
        backbone.temporal_encoder.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("sensor", out)
        ),
        backbone.image_branch.register_forward_hook(
            lambda _m, _i, out: captured.__setitem__("photo", out)
        ),
    ]
    with torch.no_grad():
        backbone.encode(force, images, idx)
        sensor, photo = model.encoded_streams(force, images, idx)
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(sensor, captured["sensor"])
    torch.testing.assert_close(photo, captured["photo"])
    fed = {}
    hook = model.memory.rnn.register_forward_hook(lambda _m, i, _o: fed.__setitem__("x", i[0]))
    with torch.no_grad():
        model.memory_states(*tensors.memory_inputs())
    hook.remove()
    block = model.memory.block
    for u in (0, 1):
        for j in range(inputs[u].n // block):
            end = torch.tensor([(j + 1) * block - 1])
            f, im, ix = tensors.windows(torch.tensor([u]), end)
            with torch.no_grad():
                s_, p_ = model.encoded_streams(f, im, ix)
            expected = torch.cat([s_.mean(1), p_.mean(1)], -1)[0]
            torch.testing.assert_close(fed["x"][u, j], expected, atol=1e-5, rtol=1e-5)
        assert not fed["x"][u, inputs[u].n // block :].any()  # unfinished blocks: padding


def test_the_encoded_memory_never_reads_a_later_step(toy_b6):
    """OmniFuser mixes all steps of a window, so a block may only be encoded once complete:
    changing any force value or photo after step t leaves the memory up to t unchanged."""
    inputs, tensors, model = _memory_case(toy_b6)
    with torch.no_grad():
        base = model.memory_states(*tensors.memory_inputs())
        for t in (20, 47, 70):
            changed = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
            front = data.INPUT_STEPS - 1
            changed.force[:, front + t + 1 :] += 4.0
            changed.index[:, front + t + 1 :] = 0  # photos after t disappear
            after = model.memory_states(*changed.memory_inputs())
            torch.testing.assert_close(after[:, : t + 1], base[:, : t + 1])
            assert not torch.allclose(after[:, t + 1 :], base[:, t + 1 :])
    assert not base[:, :15].any() and base[:, 15].abs().sum() > 0  # first block ends at 15


def test_gradients_reach_omnifusers_encoders_through_the_memory(toy_b6):
    _, tensors, model = _memory_case(toy_b6)
    model.train()
    model.memory_states(*tensors.memory_inputs()).square().sum().backward()
    parameters = dict(model.named_parameters())
    for name in (
        "backbone.temporal_embedding.value.0.weight",  # OmniFuser's force encoder
        "backbone.image_branch.encoder.projection.weight",  # its photo projection
        "memory.rnn.weight_ih_l0",
    ):
        grad = parameters[name].grad
        assert grad is not None and grad.abs().sum() > 0, name
    fusion = parameters["backbone.fusion_layers.0.project.0.weight"].grad  # not on this path
    assert fusion is None or not fusion.any()


def test_modality_dropout_reads_the_matching_memory(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:1])
    tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=SHORT_BLOCKS).eval()
    with torch.no_grad():
        full = model.memory_states(*tensors.memory_inputs())
        no_force = model.memory_states(*tensors.memory_inputs(), variant=DROP_FORCE)
        off = OmniFuserRUL(
            layout, 1, data.IMAGE_FEATURE_WIDTH, memory=SHORT_BLOCKS, use_force=False
        )
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
    # 16-step memory blocks: a toy tool (82-109 points) completes several; with the 96-step
    # default most complete none (B6 plan §14, version 6)
    spec = ArmSpec(
        players=("force", "surface"),
        epochs=8,
        seeds=(0,),
        pairs_per_epoch=2 * QUICK_PAIRS,
        memory_block=SHORT_BLOCKS.block,
    )
    folder = run_arm(
        store, steps, features, KINDS, spec, root / "runs", folds=[1], device="cpu", log=quiet
    )
    return spec, folder


def _best(folder, fold="fold-01", seed="seed-0"):
    return min(json.loads((folder / fold / seed / "fit.json").read_text())["validation_mae_s"])


def test_a_fit_learns_the_toy_rul_and_a_shuffled_control_does_not(toy_b6, toy_fits, tmp_path):
    """The fit ends below 0.7 × the constant predictor's MAE; the same fit on shuffled
    targets stays above it (and at least 1.5 × the real fit). Measured 2026-11-08 with the
    encoded memory: fit 0.44 ×, control 0.96 ×. (The margin was 0.8 until 2026-11-07; with
    versions 3-5's raw-input memory the control reached 0.78 ×.)"""
    _, store, _, features, steps, root = toy_b6
    spec, folder = toy_fits
    fold = leave_one_unit_out(store.unit_names)[0]
    validation = steps[fold.validation]
    train_y = np.concatenate([steps[u].y_s[steps[u].point] for u in fold.train])
    v = validation.point
    constant = float(
        np.average(np.abs(train_y.mean() - validation.y_s[v]), weights=validation.weight[v])
    )
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
    control = run_arm(
        store,
        shuffled,
        features,
        KINDS,
        spec,
        tmp_path / "control",
        folds=[1],
        device="cpu",
        log=quiet,
    )
    assert _best(control) > 0.7 * constant and _best(control) > 1.5 * _best(folder)


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
    assert len(record["train_loss"]) == spec.epochs and {"mse", "hinge", "slope"} <= set(
        record["train_loss"][0]
    )
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


def test_a_fine_fit_predicts_every_1s_point_and_selects_on_the_grid(toy_fine):
    _, store, _, features, steps, root = toy_fine
    spec = ArmSpec(players=("force", "surface"), epochs=3, seeds=(0,), pairs_per_epoch=QUICK_PAIRS)
    folder = run_arm(
        store, steps, features, KINDS, spec, root / "runs", folds=[1], device="cpu", log=quiet
    )
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
    spec = ArmSpec(players=("force",), epochs=2, seeds=(3,), pairs_per_epoch=QUICK_PAIRS)
    a = run_arm(
        store, steps, features, KINDS, spec, tmp_path / "a", folds=[2], device="cpu", log=quiet
    )
    b = run_arm(
        store, steps, features, KINDS, spec, tmp_path / "b", folds=[2], device="cpu", log=quiet
    )

    def read(f):
        return (
            pq.read_table(f / "fold-02" / "seed-3" / "predictions.parquet")
            .column("prediction_s")
            .to_numpy()
        )

    np.testing.assert_array_equal(read(a), read(b))


def test_inside_a_fit_the_temporal_terms_move_only_earlier_predictions(
    toy_b6, tmp_path, monkeypatch
):
    """A real fit, with a spy on every batch's loss: the temporal terms (loss − MSE) have a
    gradient on the earlier predictions only, never on the later ones, and use the spec's
    weights. (An end-to-end "fewer rises" check cannot see the run's small weights in a
    5-epoch toy fit; PROJECT_LOG.md 2026-11-07 has the measurement.)"""
    from rulfuse.omnifuser_rul import train

    _, store, _, features, steps, _ = toy_b6
    real, seen = train.pair_loss, []

    def spy(prediction, target, weight, elapsed, paired, mono_weight, slope_weight):
        loss, parts = real(prediction, target, weight, elapsed, paired, mono_weight, slope_weight)
        (grad,) = torch.autograd.grad(loss - parts["mse"], prediction, retain_graph=True)
        k = len(prediction) // 2
        seen.append(
            (float(grad[k:].abs().max()), float(grad[:k].abs().max()), mono_weight, slope_weight)
        )
        return loss, parts

    monkeypatch.setattr(train, "pair_loss", spy)
    spec = ArmSpec(players=("force", "surface"), epochs=2, seeds=(0,), pairs_per_epoch=QUICK_PAIRS)
    run_arm(store, steps, features, KINDS, spec, tmp_path, folds=[1], device="cpu", log=quiet)
    assert len(seen) == 2 * QUICK_PAIRS // (spec.batch_size // 2)  # every batch of both epochs
    assert all(later == 0.0 for later, *_ in seen)  # never on a later prediction
    assert sum(earlier > 0 for _, earlier, *_ in seen) > len(seen) / 2  # they do act
    assert {(m, w) for *_, m, w in seen} == {(spec.mono_weight, spec.slope_weight)}


def test_the_loss_defaults_and_the_cli_agree():
    import argparse

    from rulfuse import cli

    spec = ArmSpec(players=("force",))
    assert spec.mono_weight == pytest.approx(1 / 15) and spec.slope_weight == 0.05
    parser = argparse.ArgumentParser()
    cli._add_omnifuser_options(parser)
    args = parser.parse_args([])
    assert cli._arm_spec(args, ["force"]).signature() == spec.signature()  # fits match


def _pair_batch(earlier_s, later_s, y_earlier_s, y_later_s, horizon=1500.0):
    double = torch.float64
    values = torch.tensor([*earlier_s, *later_s], dtype=double) / horizon
    prediction = values.clone().requires_grad_(True)
    target = torch.tensor([*y_earlier_s, *y_later_s], dtype=double) / horizon
    elapsed = torch.tensor(y_earlier_s, dtype=double) - torch.tensor(y_later_s, dtype=double)
    paired = torch.ones(len(earlier_s), dtype=torch.bool)
    return prediction, target, torch.ones(len(target), dtype=double), elapsed / horizon, paired


def test_the_training_loss_moves_only_the_earlier_prediction():
    """pair_loss, as fit() calls it. Pair 0: the earlier prediction is stuck 450 s too low,
    the later one is at its target; pair 1: both right. The later half's gradient is the
    MSE's alone (zero at the target), so a stuck earlier prediction cannot drag it down,
    while the temporal terms lift the stuck earlier one."""
    spec = ArmSpec(players=("force",))
    prediction, target, weight, elapsed, paired = _pair_batch(
        [450.0, 900.0], [700.0, 600.0], [900.0, 900.0], [700.0, 600.0]
    )
    loss, parts = pair_loss(
        prediction, target, weight, elapsed, paired, spec.mono_weight, spec.slope_weight
    )
    (grad,) = torch.autograd.grad(loss, prediction, retain_graph=True)
    (mse_grad,) = torch.autograd.grad(parts["mse"], prediction)
    torch.testing.assert_close(grad[2:], mse_grad[2:])  # later predictions: MSE only
    torch.testing.assert_close(grad[2:], torch.zeros(2, dtype=torch.float64))  # not dragged
    assert grad[0] < mse_grad[0] < 0  # the stuck earlier one is lifted, more than by its MSE
    assert parts["hinge"].detach().item() > 0 and parts["slope"].detach().item() > 0


def test_a_stuck_later_prediction_moves_the_earlier_one_only_a_little():
    """The mirror case: the later prediction stuck 600 s too high. The earlier one settles
    about weight × H + a small slope share above its target with the run's weights, but is
    dragged all the way up with the earlier weights (1.0 and 0.5)."""

    def settled(mono, slope):
        grid = np.arange(800.0, 1400.0, 0.5)
        losses = []
        for value in grid:
            batch = _pair_batch([value], [1300.0], [900.0], [700.0])
            losses.append(pair_loss(*batch, mono, slope)[0].detach().item())
        return grid[int(np.argmin(losses))] - 900.0

    spec = ArmSpec(players=("force",))
    assert settled(spec.mono_weight, spec.slope_weight) < 160.0  # 100 s hinge + ~45 s slope
    assert settled(1.0, 0.5) > 390.0


def test_predict_unit_matches_the_training_memory(toy_b6):
    _, store, _, features, steps, _ = toy_b6
    inputs = _inputs(steps, features, store.unit_names[:2])
    tensors = data.FitTensors(inputs, data.INPUT_STEPS, "cpu")
    layout = data.Layout(steps[store.unit_names[0]].descriptors.shape[1])
    model = OmniFuserRUL(layout, 1, data.IMAGE_FEATURE_WIDTH, memory=SHORT_BLOCKS).eval()
    one = data.FitTensors(inputs[1:], data.INPUT_STEPS, "cpu")
    points = np.arange(inputs[1].n)
    # the memory of a unit does not depend on which other units share the tensors
    np.testing.assert_allclose(
        predict_unit(model, tensors, 1, points, 16),
        predict_unit(model, one, 0, points, 16),
        atol=1e-5,
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
    report = aggregate.modality_report(
        window, validation, ["force", "tool", "chip"], {u: 470.0 for u in units}
    )
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
    for unit, key, available in zip(
        images["unit"], images["key"], images["available_s"], strict=True
    ):
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


# ----------------------------------------------------------------------------- the full run


RUN_SPEC = ArmSpec(players=("force", "surface"), seeds=(0, 1), epochs=5)


def _fake_run(store, root, bias: dict[str, float], spread: float = 10.0, jitter: float = 0.0):
    """One run's fits as ``fit()`` writes them: predictions = target + the test (or
    validation) unit's bias ± ``spread`` by seed, so every number of the report is known;
    ``jitter`` adds independent noise per point (which the context mode can remove)."""
    import pyarrow as pa

    seg = store.segments
    for index, fold in enumerate(leave_one_unit_out(store.unit_names), start=1):
        for seed in RUN_SPEC.seeds:
            out = arm_dir(root, store, RUN_SPEC) / f"fold-{index:02d}" / f"seed-{seed}"
            out.mkdir(parents=True, exist_ok=True)
            rows, units, roles, values = [], [], [], []
            for role, unit in (("validation", fold.validation), ("test", fold.test)):
                mine = np.flatnonzero((seg["unit"] == unit) & seg["is_prediction_point"])
                offset = bias[unit] + (spread if seed else -spread)
                noise = np.random.default_rng([index, seed, len(rows)]).normal(0, 1, len(mine))
                rows.append(mine)
                units += [unit] * len(mine)
                roles += [role] * len(mine)
                values.append(seg["y_s"][mine] + offset + jitter * noise)
            rows_all = np.concatenate(rows)
            pq.write_table(
                pa.table(
                    {
                        "row": rows_all,
                        "unit": units,
                        "role": roles,
                        "fold": np.full(len(roles), index),
                        "seed": np.full(len(roles), seed),
                        "on_grid": np.ones(len(roles), dtype=bool),
                        "prediction_s": np.concatenate(values),
                    }
                ),
                out / "predictions.parquet",
            )
            record = {
                "fold": index,
                "seed": seed,
                "selected_epoch": 3,
                "validation_mae_s": [9.0, 8.0, 7.0, 7.5, 7.6],
                "seconds": 20.0,
                "spec": RUN_SPEC.signature(),
            }
            (out / "fit.json").write_text(json.dumps(record), encoding="utf-8")


def test_the_run_report_keeps_the_test_sealed_before_the_freeze(toy_b6, tmp_path):
    from rulfuse.omnifuser_rul.run_report import write_run_report

    _, store, *_ = toy_b6
    bias = {u: 20.0 + 5 * i for i, u in enumerate(store.unit_names)}
    _fake_run(store, tmp_path, bias)
    out = write_run_report(store, tmp_path, RUN_SPEC, today=date(2026, 11, 20))
    summary = json.loads((out / "run_summary.json").read_text())
    assert "test" not in summary and not (out / "test_per_unit.csv").exists()
    assert summary["progress"]["done"] == summary["progress"]["expected"] == 10
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Sealed until 2026-12-01" in text and "1/15 × monotonic penalty" in text
    v = {r["unit"]: r for r in summary["validation"]["per_unit"]}
    assert all(v[u]["window_mae"] == pytest.approx(bias[u]) for u in v)
    # both inference modes on the validation tools; no test numbers anywhere
    assert summary["inference"]["primary"] == "current"
    assert "Inference modes: current against context (validation tools)" in text
    assert summary["validation"]["mode_comparison"]["verdict"] == "neither"  # offsets only
    lines = (out / "validation_per_unit.csv").read_text(encoding="utf-8").strip().split("\n")
    assert lines[0].startswith("mode,")
    assert {line.split(",")[0] for line in lines[1:]} == {"current", "context"}
    roles = (out / "mode_comparison.csv").read_text(encoding="utf-8").strip().split("\n")[1:]
    assert roles and all(line.startswith("validation,") for line in roles)


def test_the_run_report_with_unseal_and_after_the_freeze(toy_b6, tmp_path):
    from rulfuse.omnifuser_rul.run_report import write_run_report

    _, store, *_ = toy_b6
    bias = {u: 20.0 + 5 * i for i, u in enumerate(store.unit_names)}
    _fake_run(store, tmp_path, bias)
    out = write_run_report(store, tmp_path, RUN_SPEC, unseal=True, today=date(2026, 11, 20))
    summary = json.loads((out / "run_summary.json").read_text())
    test = {r["unit"]: r for r in summary["test"]["per_unit"]}
    b2 = aggregate.b2_by_unit(store, "test")
    assert set(test) == set(store.unit_names)  # every tool once, as its fold's test tool
    for unit, r in test.items():
        assert r["window_mae"] == pytest.approx(bias[unit])  # the mean of the seeds
        assert r["window_bias"] == pytest.approx(bias[unit])
        assert r["B2_window_mae"] == pytest.approx(b2[unit])  # B2 on the same points
        assert r["seed_sd"] == pytest.approx(np.std([bias[unit] - 10, bias[unit] + 10], ddof=1))
        assert r["late_rate"] == 0.0  # every bias is below 100 s
    t = summary["test"]["summary"]
    gains = [b2[u] - bias[u] for u in test]
    assert t["vs_B2"]["mean_gain_s"] == pytest.approx(np.mean(gains))
    assert t["vs_B2"]["units_better"] == sum(g > 0 for g in gains)
    assert sum(b["points"] for b in t["error_by_bin"]["B6"]) == t["pooled"]["points"]
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "before the design freeze" in text and "with `--unseal`" in text
    assert "| Window MAE (s) |" in text and "### Per test tool" in text
    # constant offsets: the context mode (exact ageing) changes almost nothing
    context = {r["unit"]: r for r in summary["test"]["context"]["per_unit"]}
    assert all(abs(context[u]["window_mae"] - test[u]["window_mae"]) < 5.0 for u in test)
    assert summary["test"]["mode_comparison"]["verdict"] == "neither"
    assert "Inference modes: current against context (test tools)" in text
    trajectories = (out / "test_trajectories.csv").read_text(encoding="utf-8")
    assert "b6_context_s" in trajectories.split("\n")[0]
    assert "B6 context" in (out / "error_by_bin.csv").read_text(encoding="utf-8")
    pytest.importorskip("matplotlib")
    for name in ("rul_trajectories.png", "window_mae_per_tool.png", "error_by_rul.png"):
        assert (out / "figures" / name).stat().st_size > 10_000
    after = write_run_report(
        store, tmp_path, RUN_SPEC, today=date(2026, 12, 1), out=tmp_path / "after"
    )
    assert "with `--unseal`" not in (after / "REPORT.md").read_text(encoding="utf-8")
    assert (
        json.loads((after / "run_summary.json").read_text())["test"]["unsealed_before_freeze"]
        is False
    )


def test_the_mode_comparison_finds_a_real_difference(toy_b6, tmp_path):
    """Independent noise of 60 s per point: the context mode removes most of it, so the rule
    finds it better on every tool; REPORT.md and the CSV agree."""
    from rulfuse.omnifuser_rul.run_report import write_run_report

    _, store, *_ = toy_b6
    _fake_run(store, tmp_path, {u: 0.0 for u in store.unit_names}, spread=0.0, jitter=60.0)
    out = write_run_report(store, tmp_path, RUN_SPEC, unseal=True, today=date(2026, 11, 20))
    summary = json.loads((out / "run_summary.json").read_text())
    for role in ("validation", "test"):
        cmp = summary[role]["mode_comparison"]
        gain = cmp["window_mae_gain"]
        assert cmp["verdict"] == "context"
        assert gain["units_better"] == gain["units"] == len(store.unit_names)
        assert gain["mean_gain_s"] >= 10.0 and gain["ci95"][0] > 0
        m = cmp["metrics"]["increase_rate"]
        assert m["context"] < m["current"]  # smoother: fewer rising steps
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert text.count("**Verdict: the context mode is better.**") == 2
    rows = (out / "mode_comparison.csv").read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) == 1 + 2 * len(store.unit_names)  # validation and test, every tool
