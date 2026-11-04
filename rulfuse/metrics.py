"""Point and interval metrics (PROJECT_PLAN.md §10.2). More metrics join in Stage 4."""

from __future__ import annotations

import numpy as np

PASS_EQUIVALENT_SECONDS = 100.0  # k_sec / k = 1500 s / 15 passes (§4.3)


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(y))))


def mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction) - np.asarray(y))))


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(mse(y, prediction)))


def bias(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.asarray(prediction) - np.asarray(y)))


def mae_eq(mae_pass: float, mae_sec: float) -> float:
    """Pass-equivalent MAE: ½ (MAE in passes + MAE in seconds / 100 s)."""
    return 0.5 * (mae_pass + mae_sec / PASS_EQUIVALENT_SECONDS)


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
