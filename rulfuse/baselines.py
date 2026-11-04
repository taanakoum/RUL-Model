"""Usage-only baselines B0-B2 (PROJECT_PLAN.md §2.7, §8).

* B0: the window-aware constant k/2 (not deployable: the window is chosen with the truth);
* B1: mean training TTF − elapsed time, capped;
* B2: E[min(TTF − t, k) | TTF > t] over the training tools (cap inside the expectation, the
  MSE-optimal usage-only predictor of the capped target), with intervals from the empirical
  quantiles of min(TTF_j − t, k) over the training tools still alive at t.

Elapsed time is an *input*, so it must be causal (D14). In seconds it is the cumulative
cutting time tau. In passes it is the completed passes plus the elapsed time in the current
pass divided by the mean duration of the tool's earlier recorded passes (``DEFAULT_PASS_SECONDS``
before the first pass ends), never by the current pass's own, still unknown, duration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rulfuse import metrics as M
from rulfuse.data.folds import ToolFold
from rulfuse.data.labels import K_PASSES, K_SECONDS, ToolLabels
from rulfuse.data.timeline import DEFAULT_PASS_SECONDS


def causal_elapsed_passes(labels: ToolLabels) -> np.ndarray:
    points = labels.points
    duration = np.where(
        np.isfinite(points.mean_past_duration), points.mean_past_duration, DEFAULT_PASS_SECONDS
    )
    return points.completed_passes + np.minimum(points.elapsed_in_pass / duration, 1.0)


def b0(n: int, cap: float) -> np.ndarray:
    return np.full(n, cap / 2.0)


def b1(train_ttf: np.ndarray, elapsed: np.ndarray, cap: float) -> np.ndarray:
    return np.clip(np.mean(train_ttf) - elapsed, 0.0, cap)


def _alive_residuals(train_ttf: np.ndarray, t: float, cap: float) -> np.ndarray:
    alive = train_ttf[train_ttf > t]
    return np.minimum(alive - t, cap)


def b2(train_ttf: np.ndarray, elapsed: np.ndarray, cap: float) -> np.ndarray:
    out = np.zeros(len(elapsed))
    for index, t in enumerate(elapsed):
        residual = _alive_residuals(train_ttf, float(t), cap)
        out[index] = residual.mean() if residual.size else 0.0
    return out


def b2_interval(
    train_ttf: np.ndarray, elapsed: np.ndarray, cap: float, level: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = np.zeros(len(elapsed)), np.zeros(len(elapsed))
    tail = (1.0 - level) / 2.0
    for index, t in enumerate(elapsed):
        residual = _alive_residuals(train_ttf, float(t), cap)
        if residual.size:
            lower[index], upper[index] = np.quantile(residual, [tail, 1.0 - tail])
    return lower, upper


def score_usage_baselines(
    labels: Mapping[str, ToolLabels],
    folds: Sequence[ToolFold],
    k: float = K_PASSES,
    k_sec: float = K_SECONDS,
) -> dict:
    """B0-B2 on each fold's test tool, inside each unit's window. Returns per-fold rows and
    the mean over folds, per baseline and unit."""
    rows: list[dict] = []
    for fold in folds:
        test = labels[fold.test]
        train = [labels[tool] for tool in fold.train_tools]
        units = {
            "pass": (
                np.array([item.ttf_position for item in train], dtype=float),
                causal_elapsed_passes(test),
                test.targets.rul_pass,
                test.targets.window_pass,
                k,
            ),
            "sec": (
                np.array([item.tau_ttf for item in train], dtype=float),
                test.points.tau,
                test.targets.rul_sec,
                test.targets.window_sec,
                k_sec,
            ),
        }
        for unit, (train_ttf, elapsed, truth, window, cap) in units.items():
            x, y = elapsed[window], truth[window]
            predictions = {
                "B0": b0(len(y), cap),
                "B1": b1(train_ttf, x, cap),
                "B2": b2(train_ttf, x, cap),
            }
            lower, upper = b2_interval(train_ttf, x, cap)
            for name, prediction in predictions.items():
                row = {
                    "fold_test": fold.test,
                    "unit": unit,
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
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        entry = summary.setdefault(row["baseline"], {}).setdefault(row["unit"], {})
        for metric in ("mae", "rmse", "coverage90", "winkler90"):
            if metric in row:
                entry.setdefault(metric + "_folds", []).append(row[metric])  # type: ignore[union-attr]
    for per_unit in summary.values():
        for entry in per_unit.values():
            for key in [key for key in entry if key.endswith("_folds")]:
                entry[key.removesuffix("_folds")] = float(np.mean(entry[key]))
    return {"rows": rows, "summary": summary}
