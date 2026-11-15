"""D31: current production studies use seed 0, independently of historical ensembles."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rulfuse import paths
from rulfuse.core.folds import Fold
from rulfuse.gate3 import stack_seeds
from rulfuse.study import Study, ensemble_validation, fit_dir, read_manifest, write_manifest
from rulfuse.uncertainty import interval, mixture_moments
from tests.integration.test_artifact_integrity import prediction_rows

pytestmark = [pytest.mark.integration, pytest.mark.requirements("P14", "P15", "P20", "D31")]


@pytest.mark.parametrize("name", ["gate2a-fix", "gate2b-fix", "gate3"])
def test_current_studies_request_only_seed_zero(name):
    spec = json.loads((paths.REPO_ROOT / "configs/studies" / f"{name}.json").read_text())
    study = Study(name, spec)
    assert study.seeds == [0]
    assert all(study.arm_seeds(arm) == [0] for arm in study.arms)
    if name == "gate3":
        assert study.inner is not None
        assert study.inner["n"] == 4
        assert "gate2a-fix" in spec["inherits"]
        assert spec["decision"]["candidates"] == ["b", "c"]
    if name == "gate2b-fix":
        assert spec["requires"] == {"fusion": "F0"}


def test_single_seed_gate3_manifest_has_ten_outer_and_forty_inner_fits(tmp_path):
    study = Study(
        "single",
        {
            "seeds": [0],
            "arms": {"design": {}},
            "inner": {"n": 4, "seed": 0},
            "decision": {"type": "none"},
        },
    )
    manifest = write_manifest(study, [f"T{i}" for i in range(1, 11)], tmp_path)
    assert manifest is not None
    rows = read_manifest(manifest)
    assert len(rows) == 50
    assert {row["seeds"] for row in rows} == {"0"}
    assert sum(row["inner"] == "-1" for row in rows) == 10
    for unit in {row["fold_test"] for row in rows}:
        assert sorted(int(row["inner"]) for row in rows if row["fold_test"] == unit) == [
            -1,
            0,
            1,
            2,
            3,
        ]


def test_one_seed_retains_student_t_uncertainty_without_seed_spread(tmp_path):
    rows = prediction_rows()
    fold = Fold("T", "V", ("A", "B"))
    folder = fit_dir(tmp_path, "design", fold, 0)
    folder.mkdir(parents=True)
    pq.write_table(pa.table(rows), folder / "predictions.parquet")
    (folder / "DONE").touch()
    result = ensemble_validation(tmp_path, "design", fold, [0])
    assert result is not None
    np.testing.assert_array_equal(result["mu_s"], rows["mu_s"])
    np.testing.assert_allclose(
        result["sd_s"] ** 2, rows["sigma_s"] ** 2 * rows["nu"] / (rows["nu"] - 2)
    )
    assert len(result["seed_window_mae"]) == 1
    stacked = stack_seeds([rows])
    assert stacked["mu"].shape == (1, 2)
    mean, variance = mixture_moments(stacked["mu"], stacked["sigma"], stacked["nu"])
    np.testing.assert_allclose(mean, rows["mu_s"])
    assert (variance > 0).all()
    low, high = interval(0.9, stacked["mu"], stacked["sigma"], stacked["nu"])
    assert (low < mean).all() and (high > mean).all()


@pytest.mark.adversarial
@pytest.mark.parametrize("corruption", ["foreign-unit", "duplicate-time", "nan-prediction"])
def test_single_seed_scoring_refuses_wrong_or_corrupt_predictions(tmp_path, corruption):
    rows = prediction_rows()
    if corruption == "foreign-unit":
        rows["unit"] = np.array(["X", "X"])
    elif corruption == "duplicate-time":
        rows["tau_s"] = np.array([1.0, 1.0])
    else:
        rows["mu_s"][0] = np.nan
    fold = Fold("T", "V", ("A", "B"))
    folder = fit_dir(tmp_path, "design", fold, 0)
    folder.mkdir(parents=True)
    pq.write_table(pa.table(rows), folder / "predictions.parquet")
    (folder / "DONE").touch()
    with pytest.raises(ValueError):
        ensemble_validation(tmp_path, "design", fold, [0])
