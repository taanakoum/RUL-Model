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


def winkler(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1) -> float:
    """Mean interval score of central (1 − alpha) intervals: width plus 2/alpha × the miss."""
    y, lower, upper = np.asarray(y), np.asarray(lower), np.asarray(upper)
    score = (upper - lower) + (2 / alpha) * (
        np.maximum(lower - y, 0.0) + np.maximum(y - upper, 0.0)
    )
    return float(np.mean(score))
