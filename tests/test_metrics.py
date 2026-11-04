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
    assert M.mae_eq(2.0, 300.0) == pytest.approx(2.5)


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
