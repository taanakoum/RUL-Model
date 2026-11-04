"""P5 (core part): RUL in seconds, the cap, the window and the masks (PROJECT_PLAN.md §3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.core.targets import prediction_mask, rul_targets


def test_targets_follow_the_definitions():
    tau = np.array([10.0, 500.0, 999.0, 1000.0])
    t = rul_targets(tau, failure_s=1000.0, horizon_s=600.0)
    np.testing.assert_allclose(t.rul_s, [990.0, 500.0, 1.0, 0.0])
    np.testing.assert_allclose(t.y_s, [600.0, 500.0, 1.0, 0.0])
    np.testing.assert_array_equal(t.window, [False, True, True, True])


def test_points_after_the_failure_are_rejected_and_masked():
    with pytest.raises(ValueError):
        rul_targets(np.array([1001.0]), failure_s=1000.0, horizon_s=600.0)
    np.testing.assert_array_equal(
        prediction_mask(np.array([999.0, 1000.0, 1000.5]), 1000.0), [True, True, False]
    )
    assert not prediction_mask(np.array([1.0, 2.0]), None).any()  # censored: no targets
