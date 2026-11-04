"""P16 (part) Usage baselines B0-B2 in seconds (PROJECT_PLAN.md §2.7, §8)."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.baselines import b0, b1, b2, b2_interval, score_usage_baselines
from rulfuse.core.folds import leave_one_unit_out


def test_b2_by_hand():
    failure = np.array([10.0, 20.0, 30.0])
    # At τ = 15 the units failing at 20 and 30 are alive: residuals 5 and 15.
    assert b2(failure, np.array([15.0]), 15)[0] == pytest.approx(10.0)
    assert b2(failure, np.array([15.0]), 8)[0] == pytest.approx(6.5)  # cap inside: 5, 8
    assert b2(failure, np.array([35.0]), 15)[0] == 0.0  # nobody alive
    assert b1(failure, np.array([15.0, 25.0]), 15).tolist() == [5.0, 0.0]
    assert b0(3, 15).tolist() == [7.5, 7.5, 7.5]
    lower, upper = b2_interval(failure, np.array([15.0]), 15, level=0.9)
    assert lower[0] == pytest.approx(5.5) and upper[0] == pytest.approx(14.5)


@pytest.mark.data
def test_real_baselines_reproduce_the_plan(real_store):
    result = score_usage_baselines(real_store, leave_one_unit_out(real_store.unit_names))
    summary = result["summary"]
    assert result["horizon_s"] == 1500.0
    assert summary["B0"]["mae"] == pytest.approx(412.0, abs=0.5)  # PROJECT_PLAN.md §2.7
    assert summary["B1"]["mae"] == pytest.approx(512.5, abs=0.5)
    assert summary["B2"]["mae"] == pytest.approx(467.8, abs=0.5)
    assert summary["B2"]["coverage90"] == pytest.approx(0.50, abs=0.01)
    assert summary["B2"]["winkler90"] == pytest.approx(5388, abs=5)
