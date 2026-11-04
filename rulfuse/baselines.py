"""Usage-only baselines B0-B2 in seconds (PROJECT_PLAN.md §2.7, §8).

* B0: the window-aware constant H/2 (not deployable: the window is chosen with the truth);
* B1: mean training failure time − elapsed operating time, capped at H;
* B2: E[min(τ* − τ, H) | τ* > τ] over the training units (cap inside the expectation, the
  MSE-optimal usage-only predictor of the capped target), with intervals from the empirical
  quantiles of min(τ*_j − τ, H) over the training units still alive at τ.

Elapsed operating time τ is an input and causal by construction (D14).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rulfuse import metrics as M
from rulfuse.core.folds import Fold
from rulfuse.core.store import SegmentStore


def b0(n: int, horizon_s: float) -> np.ndarray:
    return np.full(n, horizon_s / 2.0)


def b1(train_failure: np.ndarray, tau: np.ndarray, horizon_s: float) -> np.ndarray:
    return np.clip(np.mean(train_failure) - tau, 0.0, horizon_s)


def _alive_residuals(train_failure: np.ndarray, tau: float, horizon_s: float) -> np.ndarray:
    alive = train_failure[train_failure > tau]
    return np.minimum(alive - tau, horizon_s)


def b2(train_failure: np.ndarray, tau: np.ndarray, horizon_s: float) -> np.ndarray:
    out = np.zeros(len(tau))
    for index, t in enumerate(tau):
        residual = _alive_residuals(train_failure, float(t), horizon_s)
        out[index] = residual.mean() if residual.size else 0.0
    return out


def b2_interval(
    train_failure: np.ndarray, tau: np.ndarray, horizon_s: float, level: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.zeros(len(tau)), np.zeros(len(tau))
    tail = (1.0 - level) / 2.0
    for index, t in enumerate(tau):
        residual = _alive_residuals(train_failure, float(t), horizon_s)
        if residual.size:
            lower[index], upper[index] = np.quantile(residual, [tail, 1.0 - tail])
    return lower, upper


def score_usage_baselines(store: SegmentStore, folds: Sequence[Fold]) -> dict:
    """B0-B2 on each fold's test unit, inside the window. Per-fold rows and fold means."""
    horizon = store.settings.horizon_s
    seg = store.segments
    rows: list[dict] = []
    for fold in folds:
        train_failure = np.array(
            [store.units[u]["failure_s"] for u in fold.train if store.units[u]["failure_s"]],
            dtype=float,
        )
        keep = (seg["unit"] == fold.test) & seg["window"].astype(bool)
        tau, y = seg["tau_s"][keep].astype(float), seg["rul_s"][keep].astype(float)
        predictions = {
            "B0": b0(len(y), horizon),
            "B1": b1(train_failure, tau, horizon),
            "B2": b2(train_failure, tau, horizon),
        }
        lower, upper = b2_interval(train_failure, tau, horizon)
        for name, prediction in predictions.items():
            row = {
                "fold_test": fold.test,
                "baseline": name,
                "points": int(len(y)),
                "mae": M.mae(y, prediction),
                "rmse": M.rmse(y, prediction),
                "bias": M.bias(y, prediction),
            }
            if name == "B2":
                row["coverage90"] = M.coverage(y, lower, upper)
                row["winkler90"] = M.winkler(y, lower, upper, 0.1)
            rows.append(row)
    summary: dict[str, dict[str, float]] = {}
    for name in ("B0", "B1", "B2"):
        mine = [row for row in rows if row["baseline"] == name]
        summary[name] = {
            metric: float(np.mean([row[metric] for row in mine]))
            for metric in ("mae", "rmse", "bias", "coverage90", "winkler90")
            if metric in mine[0]
        }
    return {"horizon_s": horizon, "rows": rows, "summary": summary}
