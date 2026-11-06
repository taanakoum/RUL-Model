"""Point and interval metrics, in seconds (PROJECT_PLAN.md §10.2). More join in Stage 4."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(y))))


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction) - np.asarray(y))))


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mse(y, prediction)))


def bias(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.asarray(prediction) - np.asarray(y)))


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def temporal_stability(
    prediction: np.ndarray, tau: np.ndarray, uncapped: np.ndarray | None = None
) -> dict[str, float]:
    """Over one unit's consecutive prediction points (in time order): the share of steps
    where the predicted RUL rises, the mean rise (s), and the mean |Δŷ + Δτ| (s) where the
    earlier target is below the cap (PROJECT_PLAN.md §10.2 stability: true RUL falls by Δτ)."""
    p, t = np.asarray(prediction, float), np.asarray(tau, float)
    if len(p) < 2:
        return {"increase_rate": 0.0, "mean_increase_s": 0.0, "slope_error_s": 0.0}
    dp, dt = np.diff(p), np.diff(t)
    keep = np.ones(len(dp), bool) if uncapped is None else np.asarray(uncapped, bool)[:-1]
    return {
        "increase_rate": float(np.mean(dp > 1e-6)),
        "mean_increase_s": float(np.mean(np.maximum(dp, 0.0))),
        "slope_error_s": float(np.mean(np.abs(dp[keep] + dt[keep]))) if keep.any() else 0.0,
    }


def winkler(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> float:
    """Mean interval score of central (1 − alpha) intervals: width plus 2/alpha × the miss."""
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    score = (upper - lower) + (2 / alpha) * (
        np.maximum(lower - y, 0.0) + np.maximum(y - upper, 0.0)
    )
    return float(np.mean(score))
