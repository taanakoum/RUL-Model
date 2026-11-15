"""Real synthetic components through CLI, fit, OOF calibration, reports, and freeze."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from rulfuse import cli
from rulfuse import study as study_module
from rulfuse.core.store import stable_hash
from rulfuse.gate3 import freeze
from rulfuse.study import (
    Study,
    aggregate,
    fit_dir,
    inner_fit_dir,
    read_manifest,
    run_task,
    write_manifest,
)
from rulfuse.train import fit
from tests.factories import MICRO, build_micro_bundle

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.requirements("P8", "P10", "P11", "P14", "P15", "P20", "P21", "P22", "D15"),
]


def spec_for(bundle, fusion="F1", seeds=(0,), **extra):
    return {
        "store": f"{bundle.store.dataset}/{bundle.store.signature}",
        "store_root": str(bundle.store.root),
        "features_root": str(bundle.signal.folder.parent.parent),
        "signal_encoder": "projection",
        "image_encoder": "pixelstats",
        "base": MICRO.with_overrides({"fusion": fusion}).to_dict(),
        "seeds": list(seeds),
        "folds": [bundle.folds[0].test],
        "arms": {"design": {}},
        "decision": {"type": "none"},
        **extra,
    }


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_cli_study_runs_and_regenerates_identical_validation_results(
    micro_bundle,
    tmp_path,
    monkeypatch,
    fusion,
):
    b = micro_bundle
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "cli.json").write_text(json.dumps(spec_for(b, fusion)), encoding="utf-8")
    monkeypatch.setattr(study_module, "STUDIES_DIR", configs)
    common = ["--study", "cli", "--runs-root", str(tmp_path / "runs")]
    assert cli.main(["manifest", *common]) == 0
    assert cli.main(["aggregate", *common]) == 1  # incomplete is not success
    assert cli.main(["run-task", *common, "--task", "0", "--device", "cpu"]) == 0
    assert cli.main(["aggregate", *common]) == 0
    folder = tmp_path / "runs/cli"
    first = {
        name: (folder / name).read_bytes() for name in ["summary.csv", "decision.json", "REPORT.md"]
    }
    # Poison synthetic test predictions: no validation artifact may change.
    prediction_path = fit_dir(folder, "design", b.folds[0], 0) / "predictions.parquet"
    table = pq.read_table(prediction_path)
    values = table.to_pydict()
    for i, role in enumerate(values["role"]):
        if role == "test":
            values["mu_s"][i] = 1e12
            values["rul_s"][i] = -1e12
    pq.write_table(pa.table(values), prediction_path)
    assert cli.main(["aggregate", *common]) == 0
    assert first == {name: (folder / name).read_bytes() for name in first}
    assert json.loads(first["decision.json"])["validation_only"] is True


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_outer_inner_calibration_and_frozen_hash_end_to_end(micro_bundle, tmp_path, fusion):
    b = micro_bundle
    spec = spec_for(
        b,
        fusion,
        inner={"n": 2, "seed": 0},
        decision={"type": "calibration", "arm": "design", "candidates": ["b", "c"]},
    )
    study = Study("gate", spec)
    manifest = write_manifest(study, b.store.unit_names, tmp_path)
    assert manifest is not None
    tasks = read_manifest(manifest)
    for task in range(len(tasks)):
        run_task(study, task, "cpu", tmp_path, log=lambda *_: None)
    decision = aggregate(study, tmp_path, store=b.store)
    assert decision["complete"] and decision["validation_only"]
    for k in range(2):
        folder = inner_fit_dir(study.run_dir(tmp_path), "design", b.folds[0], k, 0)
        selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        assert selection["evaluations"] == []
        rows = pq.read_table(folder / "predictions.parquet").to_pydict()
        assert set(rows["role"]) == {"heldout"}
        assert not set(rows["unit"]) & {b.folds[0].validation, b.folds[0].test}
    amendment = None if decision["constraint_met"] else "Synthetic test fixture only"
    frozen = freeze(study, tmp_path, tmp_path / "frozen.json", amended=amendment)
    hashed = {k: v for k, v in frozen.items() if k not in {"hash", "freeze_date"}}
    assert frozen["hash"] == stable_hash(hashed)
    assert frozen["calibration"] == decision["calibration_choice"]
    assert frozen["seeds"] == [0]
    assert (study.run_dir(tmp_path) / "REPORT.md").is_file()


@pytest.mark.parametrize("fusion", ["F0", "F1"])
@pytest.mark.parametrize("hi", [False, True])
def test_signal_only_transfer_dataset_fits_without_photos_or_optional_health(tmp_path, fusion, hi):
    b = build_micro_bundle(tmp_path / "data", images=False, hi=hi)
    config = MICRO.with_overrides({"fusion": fusion, "players": ("signal",)})
    result = fit(
        b.store, b.signal, None, b.folds[0], 0, config, tmp_path / "fit", log=lambda *_: None
    )
    table = pq.read_table(tmp_path / "fit/predictions.parquet").to_pydict()
    assert result["selected_step"] is not None
    assert set(table["role"]) == {"train", "validation", "test"}
    assert np.isfinite(table["mu_s"]).all() and np.isfinite(table["sigma_s"]).all()
    assert (np.asarray(table["mu_s"]) >= 0).all()
    assert (np.asarray(table["mu_s"]) <= b.store.settings.horizon_s).all()
    if not hi:
        curves = json.loads((tmp_path / "fit/curves.json").read_text(encoding="utf-8"))
        assert all(row["phase"] == "B" for row in curves)


@pytest.mark.parametrize("fusion", ["F0", "F1"])
@pytest.mark.parametrize("stop", [1, 5, 8])
def test_resume_at_phase_boundary_or_between_checkpoints_matches_full_fit(
    micro_bundle, tmp_path, fusion, stop
):
    b = micro_bundle
    config = MICRO.with_overrides({"fusion": fusion, "regularize": True, "dropout": 0.15})
    args = (b.store, b.signal, b.image, b.folds[0], 0, config)
    fit(*args, tmp_path / "full", log=lambda *_: None)
    fit(*args, tmp_path / "resume", log=lambda *_: None, stop_after_steps=stop)
    fit(*args, tmp_path / "resume", log=lambda *_: None)
    a = torch.load(tmp_path / "full/model.pt", weights_only=True)
    c = torch.load(tmp_path / "resume/model.pt", weights_only=True)
    assert all(torch.equal(a[k], c[k]) for k in a), "resume changed selected weights"
    first = pq.read_table(tmp_path / "full/predictions.parquet")
    second = pq.read_table(tmp_path / "resume/predictions.parquet")
    assert first.equals(second)
