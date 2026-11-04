"""RUL targets in seconds of operation (PROJECT_PLAN.md D3, D6, D11, §3.2).

RUL = τ* − τ at every segment end up to the failure; the training target is capped at the
horizon H; the evaluation window is RUL ≤ H. Points after the failure carry no RUL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rulfuse.core.schema import TIME_TOLERANCE_S


def prediction_mask(tau_s: np.ndarray, failure_s: float | None) -> np.ndarray:
    """Segment ends at or before the failure (none for a censored unit)."""
    tau = np.asarray(tau_s, dtype=np.float64)
    if failure_s is None:
        return np.zeros(len(tau), dtype=bool)
    return tau <= failure_s + TIME_TOLERANCE_S


@dataclass(frozen=True)
class Targets:
    rul_s: np.ndarray  # τ* − τ
    y_s: np.ndarray  # min(RUL, H): the training target
    window: np.ndarray  # RUL <= H: the evaluation window


def rul_targets(tau_s: np.ndarray, failure_s: float, horizon_s: float) -> Targets:
    """Targets of prediction points (every τ must be at or before the failure)."""
    rul = failure_s - np.asarray(tau_s, dtype=np.float64)
    if (rul < -TIME_TOLERANCE_S).any():
        raise ValueError("prediction points must not lie after the failure")
    rul = np.maximum(rul, 0.0)
    return Targets(rul, np.minimum(rul, horizon_s), rul <= horizon_s + TIME_TOLERANCE_S)
