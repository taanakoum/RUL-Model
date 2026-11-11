"""P17 (part) Stage-4 evaluation metrics (PROJECT_PLAN.md §10.2): CRPS, NLL, intervals,
Saxena prognostic metrics, stability and per-unit tables."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from rulfuse import evaluation as E


@pytest.mark.parametrize(
    ("y", "mu", "sigma", "nu"),
    [(0.7, 0.0, 1.0, 5.0), (3.0, 1.0, 2.0, 4.0), (-2.0, 0.5, 0.8, 8.0), (10.0, 0.0, 1.0, 6.0)],
)
def test_student_t_crps_matches_monte_carlo(y, mu, sigma, nu):
    rng = np.random.default_rng(0)
    x = mu + sigma * rng.standard_t(nu, size=200_000)
    half = len(x) // 2
    mc = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:half] - x[half:]))
    assert float(E.student_t_crps(y, mu, sigma, nu)) == pytest.approx(mc, rel=0.02)


def test_student_t_crps_large_nu_is_normal_closed_form():
    # normal CRPS at z = 0 is sigma * (2 phi(0) - 1/sqrt(pi)) = 0.2337 sigma
    assert E.student_t_crps(0.0, 0.0, 1.0, 1e6) == pytest.approx(
        2 * sp_stats.norm.pdf(0) - 1 / np.sqrt(np.pi), rel=1e-4
    )
    z = 1.3
    ref = z * (2 * sp_stats.norm.cdf(z) - 1) + 2 * sp_stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    assert E.student_t_crps(2 * z + 1, 1.0, 2.0, 1e6) == pytest.approx(2.0 * ref, rel=1e-4)


def test_student_t_crps_vectorized_scaling_and_validation():
    y = np.array([0.0, 1.0, -2.0])
    out = E.student_t_crps(y, 0.0, 1.0, 5.0)
    assert out.shape == (3,) and np.all(out > 0)
    np.testing.assert_allclose(E.student_t_crps(3 * y, 0.0, 3.0, 5.0), 3 * out)  # scale-equivariant
    assert out[0] < out[1] < out[2]
    with pytest.raises(ValueError):
        E.student_t_crps(0.0, 0.0, 1.0, 1.0)


def test_student_t_nll_by_hand():
    # nu = 1 is the Cauchy: log(pi (1 + z^2)) + log(sigma); z = 0.5, sigma = 2
    assert E.student_t_nll(1.0, 0.0, 2.0, 1.0) == pytest.approx(np.log(np.pi * 1.25) + np.log(2.0))
    # nu -> infinity is the normal: 0.5 log(2 pi) + log(sigma) + z^2 / 2
    assert E.student_t_nll(1.0, 0.0, 1.0, 1e8) == pytest.approx(0.5 * np.log(2 * np.pi) + 0.5)
    both = E.student_t_nll(np.array([1.0, 3.0]), 0.0, 2.0, 1.0)
    assert both == pytest.approx(
        np.mean([np.log(np.pi * 1.25) + np.log(2), np.log(np.pi * 3.25) + np.log(2)])
    )


def test_picp_mpiw_by_hand():
    y = np.array([0.0, 5.0, 10.0])
    lo, hi = np.array([1.0, 4.0, 4.0]), np.array([3.0, 6.0, 6.0])
    assert E.picp(y, lo, hi) == pytest.approx(1 / 3)
    assert E.mpiw(lo, hi) == pytest.approx(2.0)
    assert E.mpiw(np.array([0.0, 1.0]), np.array([4.0, 2.0])) == pytest.approx(2.5)


def test_interval_metrics_by_hand():
    y = np.array([0.0, 5.0, 10.0])
    lo, hi = np.array([1.0, 4.0, 4.0]), np.array([3.0, 6.0, 6.0])
    levels = (0.5, 0.8, 0.9, 0.95)
    out = E.interval_metrics(y, dict.fromkeys(levels, lo), dict.fromkeys(levels, hi))
    assert set(out) == set(levels)
    for level, alpha in zip(levels, (0.5, 0.2, 0.1, 0.05), strict=True):
        assert out[level]["coverage"] == pytest.approx(1 / 3)
        assert out[level]["width"] == pytest.approx(2.0)
        # width 2 plus 2/alpha times the mean miss (1 + 4) / 3
        assert out[level]["winkler"] == pytest.approx(2 + (2 / alpha) * 5 / 3)
    assert out[0.9]["winkler"] == pytest.approx(106 / 3)


def _life(n: int = 101, failure: float = 100.0):
    tau = np.linspace(0.0, failure, n)
    return tau, failure - tau


def test_alpha_lambda_perfect_and_biased():
    tau, y = _life()
    assert E.alpha_lambda(y, y, tau, 100.0, 40.0) == {0.25: True, 0.5: True, 0.75: True}
    # window 40..100: checks at tau 55, 70, 85 where y = 45, 30, 15 (tolerance 9, 6, 3)
    assert E.alpha_lambda(y, y + 4.0, tau, 100.0, 40.0) == {0.25: True, 0.5: True, 0.75: False}
    assert E.alpha_lambda(y, y + 10.0, tau, 100.0, 40.0) == {0.25: False, 0.5: False, 0.75: False}
    assert E.alpha_lambda(y, y - 8.0, tau, 100.0, 40.0, lambdas=(0.25,)) == {0.25: True}
    assert E.alpha_lambda(y, y + 4.0, tau, 100.0, 40.0, alpha=0.5)[0.75] is True


def test_alpha_lambda_uses_nearest_point():
    tau = np.array([0.0, 10.0, 60.0, 100.0])
    y = 100.0 - tau
    # lambda 0.5 of window 0..100 is 50: nearest point tau 60 (y = 40); prediction 47 is within 20 %
    p = np.array([0.0, 0.0, 47.0, 0.0])
    assert E.alpha_lambda(y, p, tau, 100.0, 0.0, lambdas=(0.5,)) == {0.5: True}
    p[2] = 49.0
    assert E.alpha_lambda(y, p, tau, 100.0, 0.0, lambdas=(0.5,)) == {0.5: False}


def test_prognostic_horizon_perfect_biased_and_late_convergence():
    tau, y = _life()
    assert E.prognostic_horizon(y, y, tau, 100.0, 5.0) == pytest.approx(100.0)
    assert E.prognostic_horizon(y, y + 4.0, tau, 100.0, 5.0) == pytest.approx(100.0)
    # off by 20 until tau 60, exact after: horizon is 100 - 60 = 40
    late = y + np.where(tau < 60.0, 20.0, 0.0)
    assert E.prognostic_horizon(y, late, tau, 100.0, 5.0) == pytest.approx(40.0)
    # a bad excursion mid-life resets the horizon to after it
    mid = y + np.where((tau >= 30.0) & (tau <= 50.0), 20.0, 0.0)
    assert E.prognostic_horizon(y, mid, tau, 100.0, 5.0) == pytest.approx(49.0)
    # biased beyond the bound at the last point: never inside, horizon 0
    assert E.prognostic_horizon(y, y + 10.0, tau, 100.0, 5.0) == 0.0
    last = y.copy()
    last[-1] += 50.0
    assert E.prognostic_horizon(y, last, tau, 100.0, 5.0) == 0.0


def test_monotonicity_violation_by_hand():
    tau = np.array([0.0, 10.0, 20.0])
    assert E.monotonicity_violation(100.0 - tau, tau) == 0.0
    assert E.monotonicity_violation(np.array([50.0, 50.0, 50.0]), tau) == pytest.approx(10.0)
    # steps -5 and +15 against Δτ = 10: |5| and |25|
    assert E.monotonicity_violation(np.array([50.0, 45.0, 60.0]), tau) == pytest.approx(15.0)
    assert E.monotonicity_violation(np.array([5.0]), np.array([0.0])) == 0.0


def test_point_metrics_by_hand():
    y = np.array([1000.0, 500.0, 100.0])
    p = np.array([1150.0, 350.0, 100.0])
    out = E.point_metrics(y, p, horizon_s=1500.0)
    assert out["mae"] == pytest.approx(100.0)
    assert out["mse"] == pytest.approx(15000.0)
    assert out["rmse"] == pytest.approx(np.sqrt(15000.0))
    assert out["bias"] == pytest.approx(0.0)
    assert out["mae_over_h"] == pytest.approx(100.0 / 1500.0)
    assert out["late_rate"] == pytest.approx(1 / 3)  # 150 > H/15 = 100
    assert out["early_rate"] == pytest.approx(1 / 3)
    assert out["phm_score"] == pytest.approx((np.exp(0.1 / 0.13) - 1 + np.exp(0.1 / 0.17) - 1) / 3)
    assert out["relative_accuracy"] == pytest.approx(2 / 3)  # 150 <= 200, 150 > 100, 0 <= 20
    assert set(out) == {
        "mae", "mse", "rmse", "bias", "mae_over_h", "late_rate", "early_rate", "phm_score",
        "relative_accuracy",
    }  # fmt: skip


def test_point_metrics_late_threshold_is_strict():
    y = np.array([500.0, 500.0])
    out = E.point_metrics(y, y + np.array([100.0, 100.5]), horizon_s=1500.0)
    assert out["late_rate"] == pytest.approx(0.5)  # 100 is not > 100
    assert out["early_rate"] == 0.0


def test_unit_table_mean_sd_ci():
    rows = [{"mae": 1.0, "x": np.nan}, {"mae": 3.0, "x": 2.0}, {"mae": 5.0}]
    table = E.unit_table(rows, seed=0)
    assert table["mae"]["mean"] == pytest.approx(3.0)
    assert table["mae"]["sd"] == pytest.approx(2.0)
    assert table["mae"]["n"] == 3
    assert table["mae"]["lo"] <= 3.0 <= table["mae"]["hi"]
    assert 1.0 <= table["mae"]["lo"] and table["mae"]["hi"] <= 5.0
    assert table["x"] == {"mean": 2.0, "sd": 0.0, "lo": 2.0, "hi": 2.0, "n": 1}
    assert E.unit_table(rows, seed=0) == table
    assert E.unit_table([]) == {}
