"""Gate 3 (PROJECT_PLAN.md §9, part P15): inner fits, out-of-fold calibration, the freeze."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from rulfuse.gate3 import decide_calibration, freeze, read_role_rows
from rulfuse.study import (
    Study,
    _json_value,
    aggregate,
    fit_dir,
    inner_fit_dir,
    inner_split,
    outer_selected_step,
    read_manifest,
    run_task,
    write_manifest,
)
from rulfuse.train import fit
from rulfuse.uncertainty import fit_q, kalman_rul
from tests.toy_engine import TINY, build_toy

QUIET = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return build_toy(tmp_path_factory.mktemp("toy_gate3"), n_units=6)


def _spec(toy, fold, **extra):
    return {
        "store": "toy/x",
        "store_root": str(toy.store.root),
        "features_root": str(toy.signal.folder.parent.parent),
        "signal_encoder": "projection",
        "image_encoder": "pixelstats",
        "base": TINY.to_dict(),
        "seeds": [0, 1],
        "folds": [fold.test],
        "arms": {"design": {}},
        # 4 training tools in 2 groups of two (Ruche: 8 training tools in 4 groups of two)
        "inner": {"n": 2, "seed": 0},
        "decision": {"type": "calibration", "arm": "design", "candidates": ["b", "c"]},
        **extra,
    }


def _kw(toy):
    return {
        "log": QUIET,
        "store": toy.store,
        "signal_features": toy.signal,
        "image_features": toy.image,
    }


@pytest.mark.parametrize("n", [2, 3])
def test_inner_split_never_touches_validation_or_test(toy, n):
    for fold in toy.folds:
        held_all = []
        for held, fit_units in inner_split(fold, n, 0):
            assert not {fold.test, fold.validation} & {*held, *fit_units}
            assert not set(held) & set(fit_units)
            assert set(held) | set(fit_units) == set(fold.train)
            held_all += held
        assert sorted(held_all) == sorted(fold.train)  # every training tool held out once


def test_inner_fit_reproduces_the_outer_fit_at_its_selected_step(toy, tmp_path):
    """§7.2 step 10: an inner fit trains for the outer fit's selected step with no evaluation.
    With the outer fit's own training tools, seed and step it gives the outer fit's weights."""
    fold = toy.folds[0]
    args = (toy.store, toy.signal, toy.image, fold, 0, TINY)
    outer = fit(*args, tmp_path / "outer", log=QUIET)
    steps = outer["selected_step"]
    inner = fit(
        *args,
        tmp_path / "inner",
        log=QUIET,
        heldout=[fold.validation],
        fixed_steps=steps,
    )
    assert inner["selected_step"] == steps and inner["evaluations"] == []
    a = torch.load(tmp_path / "outer" / "model.pt", weights_only=True)
    b = torch.load(tmp_path / "inner" / "model.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not (tmp_path / "inner" / "diagnostics.json").exists()
    with pytest.raises(ValueError, match="fixed_steps"):
        fit(*args, tmp_path / "bad", log=QUIET, heldout=[fold.validation])


def test_gate3_end_to_end_on_the_toy(toy, tmp_path):
    fold = toy.folds[0]
    study = Study("g3", _spec(toy, fold))
    path = write_manifest(study, toy.store.unit_names, tmp_path)
    assert path is not None
    rows = read_manifest(path)
    assert [r["inner"] for r in rows] == ["-1", "0", "1"]  # outer first, then inner
    assert aggregate(study, tmp_path, store=toy.store)["complete"] is False  # nothing run yet
    # an inner task cannot run before its outer fit: it needs the outer fit's selected step
    with pytest.raises(RuntimeError, match="outer fit is not finished"):
        run_task(study, 1, "cpu", tmp_path, **_kw(toy))
    run_task(study, 0, "cpu", tmp_path, **_kw(toy))
    for task in (1, 2):  # as on Ruche: the seeds side by side (array.sbatch --parallel-seeds)
        run_task(study, task, "cpu", tmp_path, **_kw(toy), parallel=True)
    splits = inner_split(fold, 2, 0)
    assert [len(h) for h, _ in splits] == [2, 2]
    for k, (held, fit_units) in enumerate(splits):
        for seed in (0, 1):
            folder = inner_fit_dir(tmp_path / "g3", "design", fold, k, seed)
            table = pq.read_table(folder / "predictions.parquet")
            # only the held-out tools are predicted: neither the test nor the validation tool
            assert set(table.column("role").to_pylist()) == {"heldout"}
            assert set(table.column("unit").to_pylist()) == set(held)
            info = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
            assert info["inner_held_out"] == held and info["inner_train"] == fit_units
            assert info["selected_step"] == outer_selected_step(
                study, "design", fold, seed, tmp_path
            )
    # an outer fit is not an inner fit: nothing was copied between them
    outer = fit_dir(tmp_path / "g3", "design", fold, 0)
    assert not (outer / "reused_from.txt").exists()
    decision = aggregate(study, tmp_path, store=toy.store)
    assert decision["complete"] and decision["validation_only"]
    assert decision["calibration_choice"]["method"] in ("b", "c")
    assert set(decision["mean_coverage"]) == {"raw", "a", "b", "c"}
    for m in decision["mean_coverage"].values():
        assert 0.0 <= m <= 1.0
    assert decision["adopted"] == {}  # nothing a later fit could inherit
    assert set(decision["pooled"]) == set(decision["without_thinnest"]) - {"tool", "points"}
    out = tmp_path / "g3"
    assert "Reliability" in (out / "REPORT.md").read_text(encoding="utf-8")
    assert (out / "summary.csv").is_file()
    # the freeze needs the coverage constraint (the toy's coverage may or may not meet it)
    if decision["constraint_met"]:
        frozen = freeze(study, tmp_path, tmp_path / "final.json")
    else:
        with pytest.raises(ValueError, match="coverage constraint"):
            freeze(study, tmp_path, tmp_path / "final.json")
        frozen = freeze(study, tmp_path, tmp_path / "final.json", amended="test")
        assert frozen["amended"] == "test" and frozen["constraint_met"] is False
    assert frozen["hash"] and frozen["calibration"] == decision["calibration_choice"]
    assert json.loads((tmp_path / "final.json").read_text())["design"]["max_steps"] == 24

    # B1: a second study with the same configuration copies the inner fits, not refits them
    again = Study("g3b", _spec(toy, fold))
    write_manifest(again, toy.store.unit_names, tmp_path)
    for task in range(3):
        run_task(again, task, "cpu", tmp_path, **_kw(toy))
    for k in range(2):
        copied = inner_fit_dir(tmp_path / "g3b", "design", fold, k, 0)
        assert (copied / "reused_from.txt").is_file()
        assert "inner-" in (copied / "reused_from.txt").read_text()

    # the aggregation refuses an inner fit that did not train for its outer fit's kept step
    sel = inner_fit_dir(tmp_path / "g3", "design", fold, 0, 1) / "selection.json"
    info = json.loads(sel.read_text(encoding="utf-8"))
    info["selected_step"] += 1
    sel.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="kept step"):
        aggregate(study, tmp_path, store=toy.store)


def _cell(name, points, cov, wink):
    methods = {m: {"coverage": cov[m], "winkler": wink[m], "width": 1.0} for m in cov}
    flt = {
        "coverage": 0.9,
        "winkler": 90.0,
        "window_mae_s": 10.0,
        "unfiltered_window_mae_s": 10.0,
    }
    return {
        "fold_test": name,
        "validation": name,
        "done": True,
        "points": points,
        "methods": methods,
        "filter": flt,
        "reliability": {"levels": [0.9], "b": [0.9], "c": [0.9]},
    }


def test_freeze_refuses_a_failed_coverage_constraint(tmp_path):
    """Computing the gate is not passing it (audit F04)."""
    study = Study("g3x", {"arms": {"design": {}}, "decision": {"type": "calibration"}})
    bad = {m: 0.3 for m in ("raw", "a", "b", "c")}
    wink = {m: 100.0 for m in bad}
    cells = [_cell(f"T{i}", 100, bad, wink) for i in range(10)]
    decision = decide_calibration(study, cells)
    assert decision["complete"] and decision["constraint_met"] is False
    out = tmp_path / "g3x"
    out.mkdir()
    (out / "decision.json").write_text(json.dumps(decision, default=_json_value), encoding="utf-8")
    with pytest.raises(ValueError, match="coverage constraint"):
        freeze(study, tmp_path, tmp_path / "final.json")


def test_pooled_and_without_thinnest_summaries_use_the_points(tmp_path):
    study = Study("g3y", {"arms": {"design": {}}, "decision": {"type": "calibration"}})
    good = {m: 0.9 for m in ("raw", "a", "b", "c")}
    cells = [_cell(f"T{i}", 150, good, {m: 100.0 for m in good}) for i in range(9)]
    thin = {m: 0.5 for m in good}  # a tool with 20 window points and poor coverage
    cells.append(_cell("T4", 20, thin, {m: 300.0 for m in good}))
    d = decide_calibration(study, cells)
    assert d["mean_coverage"]["b"] == pytest.approx(0.86)  # the per-tool mean: T4 weighs 1/10
    assert d["pooled"]["b"]["coverage"] == pytest.approx((9 * 150 * 0.9 + 20 * 0.5) / 1370)
    assert d["without_thinnest"]["tool"] == "T4"
    assert d["without_thinnest"]["b"]["coverage"] == pytest.approx(0.9)
    assert d["without_thinnest"]["b"]["winkler"] == pytest.approx(100.0)


def test_incomplete_study_does_not_freeze(toy, tmp_path):
    fold = toy.folds[0]
    study = Study("g3c", _spec(toy, fold))
    write_manifest(study, toy.store.unit_names, tmp_path)
    aggregate(study, tmp_path, store=toy.store)
    with pytest.raises(ValueError, match="incomplete"):
        freeze(study, tmp_path, tmp_path / "final.json")


def test_gate3_reads_validation_and_heldout_rows_only(tmp_path):
    with pytest.raises(ValueError, match="roles"):
        read_role_rows(tmp_path / "nothing.parquet", "test")


def test_fit_q_scores_only_the_masked_points():
    tau = np.arange(0.0, 60.0, 2.0)
    y = np.clip(40.0 - tau, 0.0, None)
    noisy = y + np.random.default_rng(0).normal(0, 4, len(y))
    mask = np.zeros(len(y), bool)
    mask[:10] = True
    var = np.full(len(y), 16.0)
    q_all = fit_q([(tau, y, noisy, var)], horizon_s=40.0, cap_margin_s=3.0)
    q_masked = fit_q([(tau, y, noisy, var, mask)], horizon_s=40.0, cap_margin_s=3.0)
    assert q_all > 0 and q_masked > 0
    x, _ = kalman_rul(tau, noisy, var, q_masked, 40.0, 3.0)
    assert np.all((x >= 0) & (x <= 40.0))
