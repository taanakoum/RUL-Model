"""P17 (part) Metrics (PROJECT_PLAN.md §10.2) and the resource check."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse import metrics as M
from rulfuse.resources import ResourcesBusy, check_resources


def test_point_metrics_by_hand():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 1.0])
    assert M.mae(y, p) == pytest.approx(1.0)
    assert M.mse(y, p) == pytest.approx(5 / 3)
    assert M.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert M.bias(y, p) == pytest.approx(-1 / 3)


def test_interval_metrics_by_hand():
    y = np.array([0.0, 5.0, 10.0])
    lower, upper = np.array([1.0, 4.0, 4.0]), np.array([3.0, 6.0, 6.0])
    assert M.coverage(y, lower, upper) == pytest.approx(1 / 3)
    # widths 2, 2, 2; misses 1 below and 4 above, times 2 / 0.1 = 20
    assert M.winkler(y, lower, upper, alpha=0.1) == pytest.approx((2 + 20 + 2 + 2 + 80) / 3)


def test_check_resources_raises_when_too_busy():
    status = check_resources()
    assert status.threads >= 1
    with pytest.raises(ResourcesBusy):
        check_resources(min_memory_mb=1e12)


def test_report_metrics_by_hand():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    p = np.array([250.0, 150.0, 300.0, 420.0])  # +150 (late), -50, 0, +20
    assert M.late_rate(y, p, 100.0) == pytest.approx(0.25)
    assert M.early_rate(y, p, 40.0) == pytest.approx(0.25)
    assert M.relative_accuracy(y, p, 0.2) == pytest.approx(0.5)  # 250 vs 100 and 150 vs 200 miss
    d = (p - y) / 1500.0
    expected = np.mean(np.where(d < 0, np.exp(-d / 0.17) - 1, np.exp(d / 0.13) - 1))
    assert M.phm_score(y, p, 1500.0) == pytest.approx(expected)
    assert M.phm_score(y, y + 100, 1500.0) > M.phm_score(y, y - 100, 1500.0)  # late costs more
    bins = M.error_by_bin(y, p, np.array([0.0, 250.0, 500.0]))
    assert [b["points"] for b in bins] == [2, 2]
    assert bins[0]["mae"] == pytest.approx(100.0) and bins[1]["bias"] == pytest.approx(10.0)
    lo, hi = M.bootstrap_mean_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert lo < 2.5 < hi and (lo, hi) == M.bootstrap_mean_ci(np.array([1.0, 2.0, 3.0, 4.0]))
