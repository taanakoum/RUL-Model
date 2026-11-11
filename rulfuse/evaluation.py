"""Stage-4 evaluation metrics: Student-t CRPS and NLL, interval scores, Saxena prognostic
metrics and per-unit summaries (PROJECT_PLAN.md §10.2, §10.3, §15.1 row P17)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats as sp_stats
from scipy.special import betaln

from rulfuse import metrics as M
from rulfuse.stats import unit_bootstrap

INTERVAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _t_args(y, mu, sigma, nu):
    y, mu, sigma, nu = np.broadcast_arrays(*(np.asarray(x, float) for x in (y, mu, sigma, nu)))
    return y, mu, sigma, nu


def student_t_crps(y, mu, sigma, nu) -> np.ndarray:
    """Closed-form CRPS of a location-scale Student-t (needs nu > 1), per observation.

    With z = (y − mu)/sigma and F, f the standard t cdf and pdf (Jordan, Krüger & Lerch 2019,
    scoringRules ``crps_t``; Gneiting & Raftery 2007 for the normal analogue):
    CRPS = sigma · [ z(2F(z) − 1) + 2 f(z)(nu + z²)/(nu − 1)
                     − 2 √nu B(1/2, nu − 1/2) / ((nu − 1) B(1/2, nu/2)²) ]."""
    y, mu, sigma, nu = _t_args(y, mu, sigma, nu)
    if np.any(nu <= 1):
        raise ValueError("Student-t CRPS needs nu > 1 (finite mean)")
    z = (y - mu) / sigma
    cdf, pdf = sp_stats.t.cdf(z, nu), sp_stats.t.pdf(z, nu)
    const = 2 * np.sqrt(nu) * np.exp(betaln(0.5, nu - 0.5) - 2 * betaln(0.5, nu / 2)) / (nu - 1)
    return sigma * (z * (2 * cdf - 1) + 2 * pdf * (nu + z**2) / (nu - 1) - const)


def student_t_nll(y, mu, sigma, nu) -> float:
    """Mean negative log-likelihood of a location-scale Student-t."""
    y, mu, sigma, nu = _t_args(y, mu, sigma, nu)
    z = (y - mu) / sigma
    return float(np.mean(-sp_stats.t.logpdf(z, nu) + np.log(sigma)))


def picp(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Prediction-interval coverage probability."""
    return M.coverage(y, lo, hi)


def mpiw(lo: np.ndarray, hi: np.ndarray) -> float:
    """Mean prediction-interval width."""
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def interval_metrics(
    y: np.ndarray,
    lo: Mapping[float, np.ndarray],
    hi: Mapping[float, np.ndarray],
    levels: Sequence[float] = INTERVAL_LEVELS,
) -> dict[float, dict[str, float]]:
    """Coverage, width and Winkler score (alpha = 1 − level) per central-interval level."""
    return {
        level: {
            "coverage": picp(y, lo[level], hi[level]),
            "width": mpiw(lo[level], hi[level]),
            "winkler": M.winkler(y, lo[level], hi[level], alpha=1.0 - level),
        }
        for level in levels
    }


def alpha_lambda(
    y: np.ndarray,
    prediction: np.ndarray,
    tau: np.ndarray,
    failure_tau: float,
    window_start_tau: float,
    lambdas: Sequence[float] = (0.25, 0.5, 0.75),
    alpha: float = 0.2,
) -> dict[float, bool]:
    """Alpha-lambda accuracy of one unit (Saxena et al. 2008): at the point nearest to
    window_start + λ·(failure − window_start), is |ŷ − y| <= α·y?"""
    y, p, t = (np.asarray(x, float) for x in (y, prediction, tau))
    out = {}
    for lam in lambdas:
        i = int(np.argmin(np.abs(t - (window_start_tau + lam * (failure_tau - window_start_tau)))))
        out[lam] = bool(abs(p[i] - y[i]) <= alpha * y[i])
    return out


def prognostic_horizon(
    y: np.ndarray, prediction: np.ndarray, tau: np.ndarray, failure_tau: float, bound_s: float
) -> float:
    """Earliest tau after which every prediction stays within ±bound_s of y until failure;
    returns failure_tau − that tau in seconds (0 if the last point is already outside)."""
    t = np.asarray(tau, float)
    order = np.argsort(t, kind="stable")
    ok = np.abs(np.asarray(prediction, float)[order] - np.asarray(y, float)[order]) <= bound_s
    if len(ok) == 0 or not ok[-1]:
        return 0.0
    bad = np.flatnonzero(~ok)
    first = int(bad[-1]) + 1 if len(bad) else 0
    return float(failure_tau - t[order][first])


def monotonicity_violation(prediction: np.ndarray, tau: np.ndarray) -> float:
    """Mean |Δŷ + Δτ| between consecutive points (true RUL falls by Δτ); 0 for < 2 points."""
    p, t = np.asarray(prediction, float), np.asarray(tau, float)
    if len(p) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(p) + np.diff(t))))


def point_metrics(y: np.ndarray, prediction: np.ndarray, horizon_s: float) -> dict[str, float]:
    """Point and safety metrics of §10.2 for one set of points (late/early threshold H/15)."""
    return {
        "mae": M.mae(y, prediction),
        "mse": M.mse(y, prediction),
        "rmse": M.rmse(y, prediction),
        "bias": M.bias(y, prediction),
        "mae_over_h": M.mae(y, prediction) / horizon_s,
        "late_rate": M.late_rate(y, prediction, horizon_s / 15),
        "early_rate": M.early_rate(y, prediction, horizon_s / 15),
        "phm_score": M.phm_score(y, prediction, horizon_s),
        "relative_accuracy": M.relative_accuracy(y, prediction),
    }


def unit_table(
    rows: Sequence[Mapping[str, float]], level: float = 0.95, draws: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Per metric over units: mean, sd (ddof 1) and unit-bootstrap interval; NaNs are dropped."""
    table: dict[str, dict[str, float]] = {}
    for name in dict.fromkeys(k for row in rows for k in row):
        v = np.array([row[name] for row in rows if name in row], float)
        v = v[np.isfinite(v)]
        lo, hi = unit_bootstrap(v, level=level, draws=draws, seed=seed)
        table[name] = {
            "mean": float(v.mean()) if len(v) else float("nan"),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "lo": lo,
            "hi": hi,
            "n": int(len(v)),
        }
    return table
