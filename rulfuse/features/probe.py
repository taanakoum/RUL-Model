"""Ridge probes and cross-fitted readouts (PROJECT_PLAN.md §6.2 steps 2-3, part P9).

* ``fit_ridge_cv``: standardize on the training rows, choose α by inner leave-one-unit-out
  (mean of per-unit MAE), refit on all training rows.
* ``cross_fitted_readout``: a readout of every row that never used its own unit's labels:
  rows of a training unit come from a model fitted on the *other* training units; rows of
  any other unit from the model fitted on all training units.
* ``probe_folds``: the go/no-go information check, scored on each fold's validation unit
  against the training-mean predictor; optionally with the target shuffled within each
  training unit (the control: it must collapse to about the mean predictor).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rulfuse.core.folds import Fold
from rulfuse.core.normalization import Scaler

ALPHAS = tuple(float(a) for a in 10.0 ** np.arange(-2, 6))


def _path(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alphas: Sequence[float]):
    """Predictions of ``xte`` for every α (rows standardized on ``xtr``): (len(alphas), n)."""
    scaler = Scaler.fit(xtr)
    a, b = scaler.transform(xtr), scaler.transform(xte)
    mean = float(ytr.mean())
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (ytr - mean)
    out = np.empty((len(alphas), len(xte)))
    for i, alpha in enumerate(alphas):
        coef = vt.T @ (s / (s**2 + alpha) * uty)
        out[i] = b @ coef + mean
    return out


@dataclass(frozen=True)
class Ridge:
    scaler: Scaler
    coef: np.ndarray
    intercept: float
    alpha: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(x) @ self.coef + self.intercept


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    scaler = Scaler.fit(x)
    a = scaler.transform(x)
    mean = float(y.mean())
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    coef = vt.T @ (s / (s**2 + alpha) * (u.T @ (y - mean)))
    return Ridge(scaler, coef, mean, alpha)


def select_alpha(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float] = ALPHAS
) -> float:
    groups = np.asarray(groups)
    errors = []
    for group in dict.fromkeys(groups.tolist()):
        test = groups == group
        predictions = _path(x[~test], y[~test], x[test], alphas)
        errors.append(np.abs(predictions - y[test]).mean(axis=1))
    return float(alphas[int(np.argmin(np.mean(errors, axis=0)))])


def fit_ridge_cv(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: Sequence[float] = ALPHAS
) -> Ridge:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    alpha = select_alpha(x, y, groups, alphas) if len(set(np.asarray(groups).tolist())) > 1 else alphas[len(alphas) // 2]
    return fit_ridge(x, y, alpha)


def cross_fitted_readout(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_groups: Sequence[str],
    alphas: Sequence[float] = ALPHAS,
) -> np.ndarray:
    """Readout of every row; a row's own unit's labels are never used for it. Labels of
    units outside ``train_groups`` are never used at all."""
    x, y, groups = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)
    out = np.full(len(x), np.nan)
    train = np.isin(groups, list(train_groups)) & np.isfinite(y)
    for group in train_groups:
        fit_rows = train & (groups != group)
        model = fit_ridge_cv(x[fit_rows], y[fit_rows], groups[fit_rows], alphas)
        rows = groups == group
        out[rows] = model.predict(x[rows])
    others = ~np.isin(groups, list(train_groups))
    if others.any():
        model = fit_ridge_cv(x[train], y[train], groups[train], alphas)
        out[others] = model.predict(x[others])
    return out


def shuffle_within(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y, groups = np.asarray(y, dtype=np.float64).copy(), np.asarray(groups)
    for group in dict.fromkeys(groups.tolist()):
        rows = np.flatnonzero(groups == group)
        y[rows] = y[rng.permutation(rows)]
    return y


def probe_folds(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: Sequence[Fold],
    alphas: Sequence[float] = ALPHAS,
    shuffle_seed: int | None = None,
) -> dict:
    """Per fold: validation MAE of the ridge probe and of the training-mean predictor."""
    x, y, groups = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(groups)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    rows = []
    for fold in folds:
        train = ok & np.isin(groups, fold.train)
        val = ok & (groups == fold.validation)
        if not val.any() or not train.any():
            continue
        y_train = y[train] if rng is None else shuffle_within(y[train], groups[train], rng)
        model = fit_ridge_cv(x[train], y_train, groups[train], alphas)
        mae = float(np.mean(np.abs(model.predict(x[val]) - y[val])))
        mae_mean = float(np.mean(np.abs(y_train.mean() - y[val])))
        rows.append(
            {"validation": fold.validation, "mae": mae, "mae_mean": mae_mean, "alpha": model.alpha}
        )
    mae = np.array([r["mae"] for r in rows])
    base = np.array([r["mae_mean"] for r in rows])
    return {
        "folds": rows,
        "mae": float(mae.mean()) if rows else float("nan"),
        "mae_mean_predictor": float(base.mean()) if rows else float("nan"),
        "wins": int((mae < base).sum()),
        "n_folds": len(rows),
        "beats_mean": bool(rows) and float(mae.mean()) < float(base.mean()),
    }
