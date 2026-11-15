"""Fixes of the 2026-11-15 audit (docs/AUDIT_2026-11-15.md) other than F01 (test_padded_images)
and F02/F04 (test_gate3): fit identity (F03), checkpoint random state (F05), image-free
datasets (F06), B6 seed selection (F07), the D28 disclosures (F08) and the status counter (F09).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from rulfuse.adapters import toy as toy_adapter
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import extract_signal_features
from rulfuse.features.encoders import ProjectionEncoder
from rulfuse.study import (
    Study,
    decide,
    find_identical_fit,
    render_report,
    run_task,
    write_manifest,
)
from rulfuse.tokens import build_fold_data
from rulfuse.train import StaleFitError, fit, fit_identity
from tests.toy_engine import TINY, build_toy

QUIET = lambda *_: None  # noqa: E731
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return build_toy(tmp_path_factory.mktemp("toy_audit"))


def _spec(toy, fold, **extra):
    return {
        "store": "toy/x",
        "store_root": str(toy.store.root),
        "features_root": str(toy.signal.folder.parent.parent),
        "signal_encoder": "projection",
        "image_encoder": "pixelstats",
        "base": TINY.to_dict(),
        "seeds": [0],
        "folds": [fold.test],
        "arms": {"a": {}},
        "decision": {"type": "none"},
        **extra,
    }


def _kw(toy):
    return {
        "log": QUIET,
        "store": toy.store,
        "signal_features": toy.signal,
        "image_features": toy.image,
    }


# ----------------------------------------------------------------------------- F03


def test_reuse_needs_the_same_encoders_and_code_version(toy, tmp_path):
    fold = toy.folds[0]
    first = Study("first", _spec(toy, fold))
    write_manifest(first, toy.store.unit_names, tmp_path)
    run_task(first, 0, "cpu", tmp_path, **_kw(toy))
    identity = fit_identity(TINY, fold, 0, list(fold.train))
    target = tmp_path / "other" / "a" / f"fold-{fold.test}" / "seed-0"
    same = ("projection", "pixelstats")
    source = find_identical_fit(tmp_path, "toy/x", identity, fold.test, target, same)
    assert source is not None and source.parents[2].name == "first"
    # another encoder (the audit's reproduction): not copied
    assert find_identical_fit(tmp_path, "toy/x", identity, fold.test, target, ("m", "d")) is None
    # another seed, configuration or role: not copied
    other = fit_identity(TINY.with_overrides({"lr": 3e-3}), fold, 0, list(fold.train))
    assert find_identical_fit(tmp_path, "toy/x", other, fold.test, target, same) is None
    # a fit made by other code: copied only if the study accepts that version
    sel = source / "selection.json"
    info = json.loads(sel.read_text(encoding="utf-8"))
    info["code_version"] = info["identity"]["code_version"] = 0
    sel.write_text(json.dumps(info), encoding="utf-8")
    assert find_identical_fit(tmp_path, "toy/x", identity, fold.test, target, same) is None
    assert find_identical_fit(tmp_path, "toy/x", identity, fold.test, target, same, (0,)) == source


def test_a_finished_fit_of_another_computation_is_never_skipped(toy, tmp_path):
    fold = toy.folds[0]
    study = Study("s", _spec(toy, fold))
    write_manifest(study, toy.store.unit_names, tmp_path)
    run_task(study, 0, "cpu", tmp_path, **_kw(toy))
    folder = tmp_path / "s" / "a" / f"fold-{fold.test}" / "seed-0"
    run_task(study, 0, "cpu", tmp_path, **_kw(toy))  # the same computation: skipped quietly
    # fit() on a finished folder with another seed refuses
    with pytest.raises(StaleFitError):
        fit(toy.store, toy.signal, toy.image, fold, 1, TINY, folder, log=QUIET)
    # a finished fit made by older code (F0 before the F01 fix: version 1, no identity)
    info = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    info.pop("identity")
    info["code_version"] = 0
    (folder / "selection.json").write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(StaleFitError):
        run_task(study, 0, "cpu", tmp_path, **_kw(toy))


def test_a_checkpoint_of_another_computation_is_not_resumed(toy, tmp_path):
    """F03 and F05: the checkpoint records its identity and the random states; a checkpoint of
    another computation is discarded and the fit equals a fresh one."""
    fold = toy.folds[1]
    args = (toy.store, toy.signal, toy.image, fold)
    out = tmp_path / "fit"
    assert fit(*args, 0, TINY, out, log=QUIET, stop_after_steps=12) == {"interrupted_at": 12}
    saved = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert saved["identity"] == fit_identity(TINY, fold, 0, list(fold.train))
    assert "cuda_rng" in saved and saved["cuda_rng"] is None  # a CPU fit: no CUDA state
    assert saved["torch_rng"].device.type == "cpu"
    messages = []
    fit(*args, 1, TINY, out, log=messages.append)  # seed 1 on seed 0's checkpoint
    assert any("another computation" in m for m in messages)
    fit(*args, 1, TINY, tmp_path / "fresh", log=QUIET)
    a = torch.load(out / "model.pt", weights_only=True)
    b = torch.load(tmp_path / "fresh" / "model.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


# ----------------------------------------------------------------------------- F06


def test_an_image_free_dataset_trains_and_predicts(tmp_path):
    ds = toy_adapter.build(tmp_path / "images", n_units=4)
    ds = replace(ds, units=tuple(replace(u, images=()) for u in ds.units), image_specs={})
    store = SegmentStore.open(build_segment_store(ds, tmp_path / "store"))
    for column in ("unit", "kind", "key", "path", "available_s"):
        assert column in store.images and len(store.images[column]) == 0
    fold = leave_one_unit_out(store.unit_names)[0]
    usage_only = build_fold_data(store, None, None, fold, TINY.with_overrides({"players": ()}))
    assert set(usage_only.units) == {*fold.train, fold.validation, fold.test}
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=tmp_path / "f", log=QUIET
    )
    config = TINY.with_overrides({"players": ("signal",)})
    fit(store, signal, None, fold, 0, config, tmp_path / "fit", log=QUIET)
    roles = set(pq.read_table(tmp_path / "fit" / "predictions.parquet").column("role").to_pylist())
    assert roles == {"validation", "test", "train"}
    # a store written before the fix (an image table without columns) opens the same way
    pq.write_table(pa.table({}), store.root / "images.parquet")
    assert len(SegmentStore.open(store.root).images["unit"]) == 0


# ----------------------------------------------------------------------------- F07


def _b6_fits(folder: Path, signature: dict, values: dict[int, list[float]]) -> None:
    for seed, preds in values.items():
        fit_dir = folder / "fold-01" / f"seed-{seed}"
        fit_dir.mkdir(parents=True)
        (fit_dir / "fit.json").write_text(json.dumps({"spec": signature}), encoding="utf-8")
        n = len(preds)
        pq.write_table(
            pa.table(
                {
                    "role": ["validation"] * n,
                    "fold": [1] * n,
                    "row": list(range(n)),
                    "unit": ["U1"] * n,
                    "on_grid": [True] * n,
                    "seed": [seed] * n,
                    "prediction_s": preds,
                }
            ),
            fit_dir / "predictions.parquet",
        )


def test_b6_ensembles_only_the_requested_seeds(tmp_path):
    from rulfuse.omnifuser_rul.aggregate import ensemble, load_predictions
    from rulfuse.omnifuser_rul.train import ArmSpec

    spec = ArmSpec(players=("force",), seeds=(0,))
    _b6_fits(tmp_path / "b6", spec.signature(), {0: [100.0], 1: [300.0]})
    loaded = load_predictions(tmp_path / "b6", spec.signature(), spec.seeds)
    assert np.unique(loaded["seed"]).tolist() == [0]
    assert ensemble(loaded, "validation")["prediction_s"].tolist() == [100.0]  # not 200


def test_b6_refuses_members_with_different_points(tmp_path):
    from rulfuse.omnifuser_rul.aggregate import ensemble, load_predictions
    from rulfuse.omnifuser_rul.train import ArmSpec

    spec = ArmSpec(players=("force",), seeds=(0, 1))
    _b6_fits(tmp_path / "b6", spec.signature(), {0: [100.0, 90.0], 1: [300.0]})
    with pytest.raises(ValueError, match="same points"):
        ensemble(load_predictions(tmp_path / "b6", spec.signature(), spec.seeds), "validation")


# ----------------------------------------------------------------------------- F08


def test_gate2_reports_pooled_and_without_thinnest(tmp_path):
    study = Study("g", {"arms": {"a": {}, "b": {}}, "decision": {"type": "none"}})
    rows = []
    for arm, maes in (("a", [300.0] * 9 + [1200.0]), ("b", [320.0] * 9 + [700.0])):
        for i, mae in enumerate(maes):
            points = 20 if i == 9 else 150  # the last tool is thin, like T4
            rows.append(
                {
                    "arm": arm,
                    "fold_test": f"T{i}",
                    "validation": f"V{i}",
                    "done": True,
                    "points": points,
                    "window_mae_s": mae,
                    "b2_window_mae_s": 400.0,
                    "b0_window_mae_s": 450.0,
                    "seed_sd_s": 5.0,
                    "warnings": 0,
                    "code_versions": "2",
                }
            )
    d = decide(study, rows, lambda a: None)
    assert d["means"]["a"] == pytest.approx(390.0)
    assert d["means_pooled_points"]["a"] == pytest.approx((9 * 150 * 300 + 20 * 1200) / 1370)
    assert d["means_without_thinnest"]["a"] == pytest.approx(300.0)
    assert d["thinnest_tool"] == {"tool": "V9", "points": 20}
    report = render_report(study, rows, {**d, "code_versions": [2], "mixed_code_versions": False})
    assert "Pooled-point MAE" in report and "V9 (20 window points)" in report


# ----------------------------------------------------------------------------- F09


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_status_counts_inner_fits(tmp_path):
    study = tmp_path / "runs" / "g3"
    study.mkdir(parents=True)
    (study / "manifest.tsv").write_text(
        "task\tarm\tfold_test\tseeds\tinner\n0\td\tT1\t0,1\t-1\n1\td\tT1\t0,1\t0\n",
        encoding="utf-8",
    )
    for sub in ("seed-0", "seed-1", "inner-0/seed-0", "inner-0/seed-1"):
        (study / "d" / "fold-T1" / sub).mkdir(parents=True)
        (study / "d" / "fold-T1" / sub / "DONE").write_text("ok", encoding="utf-8")
    env = {
        **os.environ,
        "RULFUSE_DATA_ROOT": str(tmp_path),
        "PYTHON": sys.executable.replace("\\", "/"),
    }
    out = subprocess.run(
        ["bash", "slurm/status.sh"], cwd=REPO, env=env, capture_output=True, text=True, timeout=120
    ).stdout
    assert "RUL-Fuse study g3: 4/4 fits" in out


def test_b6_uses_the_arm_json_seeds_by_default(tmp_path):
    from rulfuse.omnifuser_rul.aggregate import load_predictions
    from rulfuse.omnifuser_rul.train import ArmSpec

    spec = ArmSpec(players=("force",), seeds=(0,))
    _b6_fits(tmp_path / "b6", spec.signature(), {0: [100.0], 1: [300.0]})
    (tmp_path / "b6" / "arm.json").write_text(json.dumps({"spec": {"seeds": [0]}}), "utf-8")
    loaded = load_predictions(tmp_path / "b6", spec.signature())
    assert np.unique(loaded["seed"]).tolist() == [0]


def test_gate2a_style_fits_f1_copied_f0_refused(tmp_path):
    """The fits on Ruche were written before identities: no `identity`, no `code_version`.
    A rerun must copy the F1 ones (unchanged code) and retrain the F0 ones (F01)."""
    from rulfuse.config import FitConfig
    from rulfuse.train import fit_key

    fold = leave_one_unit_out([f"T{i}" for i in range(1, 11)])[0]
    spec = {"store": "nonastreda/x", "signal_encoder": "mantis8m", "image_encoder": "dinov2b"}
    (tmp_path / "gate2a").mkdir()
    (tmp_path / "gate2a" / "study.json").write_text(json.dumps(spec), encoding="utf-8")
    configs = {
        "F1-lr3e-3": FitConfig(fusion="F1", lr=3e-3),
        "F0-lr3e-3": FitConfig(fusion="F0", lr=3e-3),
    }
    for arm, config in configs.items():
        folder = tmp_path / "gate2a" / arm / f"fold-{fold.test}" / "seed-0"
        folder.mkdir(parents=True)
        legacy = {"config_key": fit_key(config), "fold_validation": fold.validation, "seed": 0}
        (folder / "selection.json").write_text(json.dumps(legacy), encoding="utf-8")
        (folder / "DONE").write_text("ok", encoding="utf-8")
    target = tmp_path / "gate2a-r2" / "x" / f"fold-{fold.test}" / "seed-0"
    enc = ("mantis8m", "dinov2b")

    def find(arm, accept=()):
        identity = fit_identity(configs[arm], fold, 0, list(fold.train))
        return find_identical_fit(
            tmp_path, "nonastreda/x", identity, fold.test, target, enc, accept
        )

    assert find("F1-lr3e-3") == tmp_path / "gate2a" / "F1-lr3e-3" / f"fold-{fold.test}" / "seed-0"
    assert find("F0-lr3e-3") is None  # trained through the F01 leak: retrained
    assert find("F0-lr3e-3", (1,)) is not None  # only if a study accepts version 1 knowingly


def test_aggregation_flags_mixed_code_versions(toy, tmp_path):
    fold = toy.folds[0]
    study = Study("mix", _spec(toy, fold, seeds=[0, 1]))
    write_manifest(study, toy.store.unit_names, tmp_path)
    run_task(study, 0, "cpu", tmp_path, **_kw(toy))
    from rulfuse.study import aggregate

    assert aggregate(study, tmp_path, store=toy.store)["mixed_code_versions"] is False
    sel = tmp_path / "mix" / "a" / f"fold-{fold.test}" / "seed-1" / "selection.json"
    info = json.loads(sel.read_text(encoding="utf-8"))
    info["code_version"] = 0
    sel.write_text(json.dumps(info), encoding="utf-8")
    decision = aggregate(study, tmp_path, store=toy.store)
    assert decision["mixed_code_versions"] is True and decision["code_versions"] == [0, 1]
    assert "are mixed" in (tmp_path / "mix" / "REPORT.md").read_text(encoding="utf-8")
