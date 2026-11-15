"""The scientific gate must execute its real checks and report their actual decisions."""

from __future__ import annotations

import json

import pytest

from rulfuse.sanity import run_checks
from tests.factories import MICRO

pytestmark = [pytest.mark.e2e, pytest.mark.requirements("P11", "P14", "P20")]


def test_sanity_runner_executes_checks_and_persists_their_actual_decisions(micro_bundle, tmp_path):
    b = micro_bundle
    checks = [2, 3, 4, 6, 7]
    report = run_checks(b.store, b.signal, b.image, MICRO, tmp_path, checks, "cpu")
    assert set(report) == {*checks, "all_pass"}
    assert report[2]["pass"] == (report[2]["validation_mae_s"] >= report[2]["b2_mae_s"] - 10)
    assert report[3]["pass"] == (abs(report[3]["validation_mae_s"] - report[3]["b2_mae_s"]) <= 20)
    assert report[4]["pass"] == (report[4]["validation_mae_s"] < 0.8 * report[4]["b2_mae_s"])
    assert report[6]["pass"] and report[6]["max_abs_difference_s"] < 1e-3
    assert report[7] == {"identical": True, "pass": True}
    assert report["all_pass"] == all(report[k]["pass"] for k in checks)
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved == {str(k): v for k, v in report.items()}
    # These are real fits of the active seed, rather than mocked metric reports.
    for check in [2, 3, 4]:
        selection = json.loads((tmp_path / f"check-{check}/selection.json").read_text())
        assert selection["seed"] == 0 and selection["selected_step"] is not None


def test_overfit_and_toy_gates_report_the_fitted_model_and_selected_coverage(
    micro_bundle,
    tmp_path,
):
    b = micro_bundle
    report = run_checks(b.store, b.signal, b.image, MICRO, tmp_path, [1, 5], "cpu")
    assert report[1]["pass"] == (report[1]["train_mae_s"] < 10)
    overfit = json.loads((tmp_path / "check-1/selection.json").read_text())
    config = json.loads((tmp_path / "check-1/config.json").read_text())
    assert len(overfit["identity"]["train"]) == 1
    assert config["regularize"] is False and config["dropout"] == 0
    assert config["weight_decay"] == 0 and config["phase_a"] is False
    selection = json.loads((tmp_path / "check-5/selection.json").read_text())
    # Coverage belongs to the restored model, not whichever evaluation happened last.
    selected_coverage = selection["evaluations"][selection["selected_eval"]]["coverage90"]
    assert report[5]["coverage90"] == selected_coverage
    assert report[5]["pass"] == (
        report[5]["validation_mae_s"] < report[5]["b2_mae_s"] and 0.8 <= selected_coverage <= 0.98
    )
