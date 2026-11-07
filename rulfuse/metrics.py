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


def phm_score(y: np.ndarray, prediction: np.ndarray, horizon_s: float) -> float:
    """Mean asymmetric score of PROJECT_PLAN.md §10.2: d = (ŷ − y)/H; e^(−d/0.17) − 1 for an
    early prediction (d < 0), e^(d/0.13) − 1 for a late one, which is the costlier mistake."""
    d = (np.asarray(prediction, float) - np.asarray(y, float)) / horizon_s
    return float(np.mean(np.where(d < 0, np.exp(-d / 0.17) - 1.0, np.exp(d / 0.13) - 1.0)))


def late_rate(y: np.ndarray, prediction: np.ndarray, threshold_s: float) -> float:
    """Share of points predicted more than ``threshold_s`` too optimistic (ŷ − y > threshold)."""
    return float(np.mean(np.asarray(prediction, float) - np.asarray(y, float) > threshold_s))


def early_rate(y: np.ndarray, prediction: np.ndarray, threshold_s: float) -> float:
    """Share of points predicted more than ``threshold_s`` too pessimistic."""
    return float(np.mean(np.asarray(y, float) - np.asarray(prediction, float) > threshold_s))


def relative_accuracy(y: np.ndarray, prediction: np.ndarray, tolerance: float = 0.2) -> float:
    """Share of points (true RUL > 0) whose prediction is within ±tolerance of the true RUL."""
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    keep = y > 0
    return float(np.mean(np.abs(p[keep] - y[keep]) <= tolerance * y[keep])) if keep.any() else 0.0


def error_by_bin(
    y: np.ndarray, prediction: np.ndarray, edges: np.ndarray
) -> list[dict[str, float]]:
    """MAE and bias per true-RUL bin [edges[i], edges[i+1]) (the last bin is closed)."""
    y, p = np.asarray(y, float), np.asarray(prediction, float)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = i == len(edges) - 2
        keep = (y >= lo) & ((y <= hi) if last else (y < hi))
        rows.append(
            {
                "from_s": float(lo),
                "to_s": float(hi),
                "points": int(keep.sum()),
                "mae": mae(y[keep], p[keep]) if keep.any() else float("nan"),
                "bias": bias(y[keep], p[keep]) if keep.any() else float("nan"),
            }
        )
    return rows


def bootstrap_mean_ci(
    values: np.ndarray, level: float = 0.95, draws: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval of the mean over resampled units (with replacement)."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return (float(v.mean()), float(v.mean())) if len(v) else (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(draws, len(v)))].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def winkler(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> float:
    """Mean interval score of central (1 − alpha) intervals: width plus 2/alpha × the miss."""
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    score = (upper - lower) + (2 / alpha) * (
        np.maximum(lower - y, 0.0) + np.maximum(y - upper, 0.0)
    )
    return float(np.mean(score))
