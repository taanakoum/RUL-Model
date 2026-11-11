"""Uncertainty and calibration of the seed ensemble, in seconds.

Implements PROJECT_PLAN.md §9 (predictive mixture, calibration methods a/b/c, causal Kalman RUL
filter) and §7.2 steps 10-12, for part P15 of the test plan (§15.1). numpy/scipy only.

Conventions: arrays of per-seed Student-t parameters have shape (S, N) (S seed models, N
prediction points), nu > 2, all quantities in seconds, RUL in [0, H].
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence

import numpy as np
from scipy.special import stdtr

_BRACKET_SIGMAS = 50.0
_MAX_BISECT = 200


def _params(
    mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.atleast_2d(np.asarray(mu, float))
    sigma = np.broadcast_to(np.atleast_2d(np.asarray(sigma, float)), mu.shape)
    nu = np.broadcast_to(np.atleast_2d(np.asarray(nu, float)), mu.shape)
    return mu, sigma, nu


def mixture_moments(
    mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the equal-weight mixture of S Student-t members (§9.1).

    mean = avg mu_s; variance = avg(sigma_s^2 nu_s / (nu_s - 2)) + var(mu_s) (population
    variance over seeds). Inputs (S, N); outputs (N,)."""
    mu, sigma, nu = _params(mu, sigma, nu)
    if np.any(nu <= 2.0):
        raise ValueError("mixture_moments needs nu > 2")
    aleatoric = np.mean(sigma**2 * nu / (nu - 2.0), axis=0)
    return mu.mean(axis=0), aleatoric + mu.var(axis=0)


def mixture_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """CDF of the equal-weight mixture of location-scale Student-t members at x.

    x is (N,) (or scalar), or (Q, N) for several evaluation points per prediction point;
    mu, sigma, nu are (S, N). Returns the shape of x."""
    mu, sigma, nu = _params(mu, sigma, nu)
    x = np.asarray(x, float)
    xe = x[..., None, :] if x.ndim else x
    z = (xe - mu) / sigma  # (..., S, N)
    return stdtr(nu, z).mean(axis=-2)  # Student-t CDF (scipy.special.stdtr == scipy.stats.t.cdf)


def mixture_quantiles(
    q: Sequence[float] | np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    horizon_s: float | None = None,
    tol: float = 1e-3,
) -> np.ndarray:
    """Quantiles of the mixture by vectorized bisection on its CDF (§9.1). Returns (len(q), N).

    The bracket is [min_s(mu - 50 sigma), max_s(mu + 50 sigma)] per point; bisection stops when
    every bracket is narrower than tol (seconds). With horizon_s the results are clipped to
    [0, horizon_s]."""
    mu, sigma, nu = _params(mu, sigma, nu)
    qs = np.atleast_1d(np.asarray(q, float))
    n = mu.shape[1]
    lo = np.broadcast_to((mu - _BRACKET_SIGMAS * sigma).min(axis=0), (len(qs), n)).copy()
    hi = np.broadcast_to((mu + _BRACKET_SIGMAS * sigma).max(axis=0), (len(qs), n)).copy()
    target = qs[:, None]
    for _ in range(_MAX_BISECT):
        if np.max(hi - lo) <= tol:
            break
        mid = 0.5 * (lo + hi)
        below = mixture_cdf(mid, mu, sigma, nu) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    out = 0.5 * (lo + hi)
    if horizon_s is not None:
        out = np.clip(out, 0.0, float(horizon_s))
    return out


def interval(
    level: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    horizon_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Central `level` interval (lower, upper) of the mixture, clipped to [0, H] if given."""
    alpha = 1.0 - level
    lo, hi = mixture_quantiles([alpha / 2.0, 1.0 - alpha / 2.0], mu, sigma, nu, horizon_s)
    return lo, hi


# ---------------------------------------------------------------------------------------------
# Calibration (§9.2): one scale c applied to every member's sigma
# ---------------------------------------------------------------------------------------------


def apply_scale(sigma: np.ndarray, c: float) -> np.ndarray:
    """Scale every member's sigma by c."""
    return np.asarray(sigma, float) * float(c)


def _coverage_at_scale(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    c: float,
    level: float,
    horizon_s: float | None,
) -> float:
    # y lies in the central interval iff its mixture PIT is in [alpha/2, 1 - alpha/2]; this is
    # exact, needs no bisection, and equals the coverage of the interval clipped to [0, H] for
    # y in [0, H] (outside that range a clipped interval cannot contain y, handled below).
    alpha = 1.0 - level
    pit = mixture_cdf(y, mu, sigma * c, nu)
    inside = (pit >= alpha / 2.0) & (pit <= 1.0 - alpha / 2.0)
    if horizon_s is not None:
        inside &= (y >= 0.0) & (y <= horizon_s)
    return float(np.mean(inside))


def fit_scale(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: np.ndarray,
    level: float = 0.9,
    horizon_s: float | None = None,
    n_iter: int = 40,
) -> float:
    """The scale c such that the central `level` mixture interval of (mu, c sigma, nu) covers
    `level` of y (§9.2 methods a and b: same function, different data).

    Coverage is monotone in c; bisection on log c in [1e-2, 1e2]. Returns the smallest c
    reaching the target, or the bracket end if the target is not reachable."""
    y = np.asarray(y, float)
    mu, sigma, nu = _params(mu, sigma, nu)

    def cov(c: float) -> float:
        return _coverage_at_scale(y, mu, sigma, nu, c, level, horizon_s)

    lo, hi = math.log(1e-2), math.log(1e2)
    if cov(math.exp(lo)) >= level:
        return math.exp(lo)
    if cov(math.exp(hi)) < level:
        return math.exp(hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if cov(math.exp(mid)) >= level:
            hi = mid
        else:
            lo = mid
    return math.exp(hi)


def conformal_scale(
    y: np.ndarray, mu_mix: np.ndarray, sd_mix: np.ndarray, level: float = 0.9
) -> float:
    """Method (c): the finite-sample-corrected `level` quantile q of |y - mu| / sd, i.e. the
    ceil((n + 1) level)-th smallest score. Intervals are mu +/- q sd (see conformal_interval).
    Returns inf if (n + 1) level > n (too few points for the level)."""
    score = np.abs(np.asarray(y, float) - np.asarray(mu_mix, float)) / np.asarray(sd_mix, float)
    n = score.size
    k = math.ceil(round((n + 1) * level, 9))
    if k > n:
        return math.inf
    return float(np.sort(score.ravel())[k - 1])


def conformal_interval(
    mu_mix: np.ndarray, sd_mix: np.ndarray, q: float, horizon_s: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """mu +/- q sd, clipped to [0, H] if given."""
    mu_mix, sd_mix = np.asarray(mu_mix, float), np.asarray(sd_mix, float)
    lower, upper = mu_mix - q * sd_mix, mu_mix + q * sd_mix
    if horizon_s is not None:
        lower, upper = np.clip(lower, 0.0, horizon_s), np.clip(upper, 0.0, horizon_s)
    return lower, upper


def inner_folds(
    train_units: Sequence[Hashable], n_inner: int = 4, seed: int = 0
) -> list[tuple[list[Hashable], list[Hashable]]]:
    """Split an outer fold's training units into n_inner disjoint groups (sizes as equal as
    possible, deterministic by seed). Returns [(held_out_units, fit_units), ...]; every unit is
    held out exactly once and nothing outside train_units appears (§7.2 step 10)."""
    units = sorted(dict.fromkeys(train_units), key=str)
    if not 2 <= n_inner <= len(units):
        raise ValueError(f"need 2 <= n_inner <= {len(units)} training units, got {n_inner}")
    order = np.random.default_rng(seed).permutation(len(units))
    groups = [[units[i] for i in idx] for idx in np.array_split(order, n_inner)]
    return [
        (g, [u for k, h in enumerate(groups) if k != i for u in h]) for i, g in enumerate(groups)
    ]


# ---------------------------------------------------------------------------------------------
# Causal Kalman RUL filter (§9.3)
# ---------------------------------------------------------------------------------------------


def kalman_rul(
    tau: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    q: float,
    horizon_s: float,
    cap_margin_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal Kalman filter of one unit's RUL predictions, in time order.

    State x = RUL; predict x_{j+1} = x_j - dtau_j with P += q dtau_j (q in s^2/s); update with
    the measurement mu_j of variance var_j. A measurement at the cap (mu_j >= horizon_s -
    cap_margin_s) is output unfiltered and resets the state to (mu_j, var_j); the first point
    initialises the state the same way. Output means are clipped to [0, horizon_s]. Returns
    (filtered_mean, filtered_var); output j depends only on inputs <= j."""
    tau, mu, var = (np.asarray(a, float).ravel() for a in (tau, mu, var))
    if not (tau.shape == mu.shape == var.shape):
        raise ValueError("tau, mu, var must have the same length")
    if np.any(np.diff(tau) < 0):
        raise ValueError("tau must be non-decreasing")
    n = mu.size
    mean, pvar = np.empty(n), np.empty(n)
    cap = horizon_s - cap_margin_s
    x = p = 0.0
    for j in range(n):
        vj = max(var[j], 1e-12)
        if j == 0 or mu[j] >= cap:
            x, p = mu[j], vj
        else:
            dt = tau[j] - tau[j - 1]
            x_pred, p_pred = x - dt, p + q * dt
            gain = p_pred / (p_pred + vj)
            x, p = x_pred + gain * (mu[j] - x_pred), (1.0 - gain) * p_pred
        mean[j], pvar[j] = np.clip(x, 0.0, horizon_s), p
    return mean, pvar


def fit_q(
    units: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    grid: np.ndarray | None = None,
    *,
    horizon_s: float,
    cap_margin_s: float,
) -> float:
    """Choose q from a log grid (default np.logspace(-3, 3, 25), s^2/s) minimising the mean over
    units of the MAE of the filtered mean against y. `units` are (tau, y, mu, var) tuples of
    out-of-fold predictions, each in time order (§9.3). Ties go to the smallest q."""
    qs = np.logspace(-3, 3, 25) if grid is None else np.asarray(grid, float)
    scores = [
        np.mean(
            [
                np.mean(np.abs(kalman_rul(tau, mu, var, float(q), horizon_s, cap_margin_s)[0] - y))
                for tau, y, mu, var in units
            ]
        )
        for q in qs
    ]
    return float(qs[int(np.argmin(scores))])
