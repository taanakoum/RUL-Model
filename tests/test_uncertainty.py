"""Part P15: uncertainty and calibration (PROJECT_PLAN.md §9, §15.1 row P15)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rulfuse.uncertainty import (
    apply_scale,
    conformal_interval,
    conformal_scale,
    fit_q,
    fit_scale,
    inner_folds,
    interval,
    kalman_rul,
    mixture_cdf,
    mixture_moments,
    mixture_quantiles,
)

H = 1500.0


def _members() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.array([[600.0, 900.0], [650.0, 880.0], [560.0, 950.0]])
    sigma = np.array([[40.0, 60.0], [50.0, 55.0], [45.0, 70.0]])
    nu = np.array([[5.0, 8.0], [6.0, 4.0], [10.0, 7.0]])
    return mu, sigma, nu


def _draw(mu, sigma, nu, n=200_000, seed=0) -> np.ndarray:
    """Monte Carlo draws of the mixture, (n, N)."""
    rng = np.random.default_rng(seed)
    s, npts = mu.shape
    member = rng.integers(0, s, size=(n, npts))
    cols = np.arange(npts)[None, :]
    return mu[member, cols] + sigma[member, cols] * rng.standard_t(nu[member, cols])


def test_moments_match_monte_carlo():
    mu, sigma, nu = _members()
    mean, var = mixture_moments(mu, sigma, nu)
    x = _draw(mu, sigma, nu)
    np.testing.assert_allclose(mean, x.mean(axis=0), rtol=0.01)
    np.testing.assert_allclose(var, x.var(axis=0), rtol=0.1)  # heavy tails: noisy MC variance
    expected = np.mean(sigma**2 * nu / (nu - 2), axis=0) + mu.var(axis=0)
    np.testing.assert_allclose(var, expected)


def test_quantiles_match_monte_carlo():
    mu, sigma, nu = _members()
    qs = np.array([0.05, 0.5, 0.95])
    got = mixture_quantiles(qs, mu, sigma, nu)
    assert got.shape == (3, 2)
    x = _draw(mu, sigma, nu)
    np.testing.assert_allclose(got, np.quantile(x, qs, axis=0), rtol=0.01)
    # the CDF at the returned quantiles recovers the levels
    for k, p in enumerate(qs):
        np.testing.assert_allclose(mixture_cdf(got[k], mu, sigma, nu), p, atol=1e-5)


def test_single_member_equals_t_quantile():
    mu, sigma, nu = np.array([[700.0]]), np.array([[30.0]]), np.array([[5.0]])
    qs = [0.025, 0.5, 0.975]
    got = mixture_quantiles(qs, mu, sigma, nu, tol=1e-6)[:, 0]
    np.testing.assert_allclose(got, stats.t.ppf(qs, 5.0, loc=700.0, scale=30.0), atol=1e-4)


def test_quantiles_clipped_and_interval():
    mu, sigma, nu = np.array([[20.0, 1490.0]]), np.array([[50.0, 50.0]]), np.array([[4.0, 4.0]])
    lo, hi = interval(0.9, mu, sigma, nu, horizon_s=H)
    assert lo[0] == 0.0 and hi[1] == H
    assert np.all(lo <= hi)
    lo_u, hi_u = interval(0.9, mu, sigma, nu)
    assert lo_u[0] < 0.0 and hi_u[1] > H


def test_fit_scale_recovers_variance_inflation():
    rng = np.random.default_rng(1)
    n, s = 3000, 3
    mu_true = rng.uniform(200, 1300, n)
    sigma = np.full((s, n), 40.0)
    nu = np.full((s, n), 6.0)
    mu = mu_true + rng.normal(0, 1, (s, n)) * 1.0
    y = mu_true + 2.0 * 40.0 * rng.standard_t(6.0, n)  # true scale twice the model's
    c = fit_scale(y, mu, sigma, nu, level=0.9)
    assert abs(c - 2.0) / 2.0 < 0.1
    lo, hi = interval(0.9, mu, apply_scale(sigma, c), nu)
    assert abs(np.mean((y >= lo) & (y <= hi)) - 0.9) < 0.01


def test_fit_scale_bracket_ends():
    mu, sigma, nu = np.full((1, 50), 500.0), np.full((1, 50), 1.0), np.full((1, 50), 5.0)
    assert fit_scale(np.full(50, 500.0), mu, sigma, nu) == pytest.approx(1e-2)
    assert fit_scale(np.full(50, 5000.0), mu, sigma, nu) == pytest.approx(1e2)


def test_conformal_coverage_exchangeable():
    rng = np.random.default_rng(2)

    def sample(n):
        sd = rng.uniform(20, 80, n)
        mu = rng.uniform(100, 1400, n)
        return mu + 1.7 * sd * rng.normal(size=n), mu, sd

    y, mu, sd = sample(2000)
    q = conformal_scale(y, mu, sd, level=0.9)
    y2, mu2, sd2 = sample(20000)
    lo, hi = conformal_interval(mu2, sd2, q, horizon_s=H)
    assert abs(np.mean((y2 >= lo) & (y2 <= hi)) - 0.9) < 0.03
    assert q == pytest.approx(1.7 * stats.norm.ppf(0.95), rel=0.1)


def test_conformal_finite_sample_rank():
    y, mu, sd = np.arange(1.0, 10.0), np.zeros(9), np.ones(9)  # scores 1..9, n = 9
    assert conformal_scale(y, mu, sd, level=0.9) == 9.0  # ceil(10 * 0.9) = 9 -> 9th smallest
    assert conformal_scale(y, mu, sd, level=0.95) == np.inf  # ceil(9.5) = 10 > n
    y, mu, sd = np.arange(1.0, 20.0), np.zeros(19), np.ones(19)  # n = 19: ceil(18) = 18
    assert conformal_scale(y, mu, sd, level=0.9) == 18.0


def test_inner_folds_disjoint_cover_and_deterministic():
    train = [f"T{i}" for i in (1, 2, 3, 5, 6, 7, 8, 10)]
    folds = inner_folds(train, n_inner=4, seed=3)
    assert len(folds) == 4
    held = [u for h, _ in folds for u in h]
    assert sorted(map(str, held)) == sorted(map(str, train)) and len(set(held)) == len(held)
    for h, fit in folds:
        assert len(h) == 2 and len(fit) == 6
        assert not set(h) & set(fit)
        assert set(h) | set(fit) == set(train)
    assert inner_folds(train, 4, 3) == folds
    assert inner_folds(train, 4, 4) != folds
    uneven = inner_folds(list(range(7)), 4, 0)
    assert sorted(len(h) for h, _ in uneven) == [1, 2, 2, 2]


def _synthetic_unit(rng, n=120, dt=10.0, sd=60.0, start=1300.0):
    tau = np.arange(n) * dt
    y = start - tau
    return tau, y, y + rng.normal(0, sd, n), np.full(n, sd**2)


def test_kalman_reduces_error_and_fit_q_picks_small_q():
    rng = np.random.default_rng(4)
    tau, y, mu, var = _synthetic_unit(rng)
    mean, pvar = kalman_rul(tau, mu, var, q=0.01, horizon_s=H, cap_margin_s=10.0)
    assert np.mean(np.abs(mean - y)) < 0.6 * np.mean(np.abs(mu - y))
    assert np.all(pvar > 0) and np.all((mean >= 0) & (mean <= H))
    units = [_synthetic_unit(rng) for _ in range(4)]
    q = fit_q(units, horizon_s=H, cap_margin_s=10.0)
    assert q <= 1.0  # the truth has slope exactly -1, so little process noise is needed
    assert q in np.logspace(-3, 3, 25)


def test_kalman_is_causal():
    rng = np.random.default_rng(5)
    tau, y, mu, var = _synthetic_unit(rng)
    base, base_var = kalman_rul(tau, mu, var, 0.1, H, 10.0)
    mu2, var2 = mu.copy(), var.copy()
    mu2[70:] += rng.normal(0, 500, mu2[70:].size)
    var2[70:] *= 3.0
    out, out_var = kalman_rul(tau, mu2, var2, 0.1, H, 10.0)
    np.testing.assert_array_equal(out[:70], base[:70])
    np.testing.assert_array_equal(out_var[:70], base_var[:70])
    assert not np.allclose(out[70:], base[70:])


def test_kalman_cap_passthrough_and_reset():
    tau = np.arange(8) * 10.0
    mu = np.array([1500.0, 1500.0, 1495.0, 900.0, 1500.0, 700.0, 690.0, 680.0])
    var = np.full(8, 100.0**2)
    mean, pvar = kalman_rul(tau, mu, var, 1.0, H, cap_margin_s=10.0)
    at_cap = mu >= H - 10.0
    np.testing.assert_array_equal(mean[at_cap], mu[at_cap])
    np.testing.assert_array_equal(pvar[at_cap], var[at_cap])
    assert mean[3] != mu[3]  # the first uncapped point after a capped one is filtered
    # the state resets at the cap: earlier history cannot influence later outputs
    mu2 = mu.copy()
    mu2[:4] = [1000.0, 1100.0, 1200.0, 300.0]
    out, _ = kalman_rul(tau, mu2, var, 1.0, H, 10.0)
    np.testing.assert_array_equal(out[4:], mean[4:])
