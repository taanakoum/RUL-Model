"""Stage 7-8 tools: Shapley arithmetic and the test seal of ``rulfuse.final`` (D15)."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from rulfuse import final
from rulfuse.core.store import stable_hash
from rulfuse.shapley import coalition_key, interaction, per_tool_values, shapley_values

pytestmark = [pytest.mark.unit, pytest.mark.requirements("P15", "P20")]


def test_two_player_closed_form():
    v = {(): -400.0, ("F",): -380.0, ("T",): -350.0, ("F", "T"): -330.0}
    phi = shapley_values(v, ["F", "T"])
    assert phi["F"] == pytest.approx(0.5 * (20.0 + 20.0))
    assert phi["T"] == pytest.approx(0.5 * (50.0 + 50.0))
    assert interaction(v, "F", "T", ["F", "T"]) == pytest.approx(-330 + 380 + 350 - 400)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_efficiency(n):
    rng = np.random.default_rng(n)
    players = [f"P{i}" for i in range(n)]
    from itertools import combinations

    v = {
        coalition_key(c): float(-rng.uniform(200, 500))
        for r in range(n + 1)
        for c in combinations(players, r)
    }
    phi = shapley_values(v, players)
    assert sum(phi.values()) == pytest.approx(v[coalition_key(players)] - v[()])


def test_missing_coalition_refused():
    with pytest.raises(ValueError, match="without a value"):
        shapley_values({(): 0.0, ("F",): 1.0}, ["F", "T"])


def test_per_tool_arrays_in_mae_units():
    mae = {
        (): np.array([400.0, 500.0]),
        ("F",): np.array([390.0, 450.0]),
        ("T",): np.array([380.0, 520.0]),
        ("F", "T"): np.array([350.0, 430.0]),
    }
    out = per_tool_values(mae, ["F", "T"])
    assert out["gain_over_empty"].tolist() == [50.0, 70.0]
    total = out["phi"]["F"] + out["phi"]["T"]
    assert total.tolist() == pytest.approx([50.0, 70.0])
    assert out["leave_one_out"]["F"].tolist() == [30.0, 90.0]  # MAE without F minus MAE with
    assert out["add_one"]["T"].tolist() == [20.0, -20.0]  # T alone vs B2


def _frozen(tmp_path, freeze_date="2026-12-01"):
    body = {"design": {"fusion": "F0"}, "calibration": {"method": "b", "level": 0.9}}
    frozen = {**body, "hash": stable_hash(body), "freeze_date": freeze_date}
    path = tmp_path / "final.json"
    path.write_text(json.dumps(frozen))
    return path


def test_seal_refuses_without_freeze_file(tmp_path):
    with pytest.raises(final.SealedError, match="does not exist"):
        final.check_seal(tmp_path / "final.json", today=date(2026, 12, 5))


def test_seal_opens_once_frozen_whatever_the_date(tmp_path):
    path = _frozen(tmp_path)
    assert final.check_seal(path, today=date(2026, 11, 20))["design"] == {"fusion": "F0"}


def test_seal_refuses_edited_freeze_file(tmp_path):
    path = _frozen(tmp_path)
    data = json.loads(path.read_text())
    data["design"]["fusion"] = "F1"
    path.write_text(json.dumps(data))
    with pytest.raises(final.SealedError, match="hash mismatch"):
        final.check_seal(path, today=date(2026, 12, 5))
