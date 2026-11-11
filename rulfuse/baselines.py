"""Usage-only baselines B0-B2 in seconds (PROJECT_PLAN.md §2.7, §8).

* B0: the window-aware constant H/2 (not deployable: the window is chosen with the truth);
* B1: mean training failure time − elapsed operating time, capped at H;
* B2: E[min(τ* − τ, H) | τ* > τ] over the training units (cap inside the expectation, the
  MSE-optimal usage-only predictor of the capped target), with intervals from the empirical
  quantiles of min(τ*_j − τ, H) over the training units still alive at τ.

Elapsed operating time τ is an input and causal by construction (D14).

Feature baselines B4, B5 and the reference R1 (§8; part P16, §15.1) are further down:

* B4: ridge on the level descriptors (absolute and relative to the unit's causal baseline,
  mean of the last 3 segments) and the usage features, against the capped RUL;
* B5: an image HI readout (cross-fitted ridge) and the degradation extrapolation
  (:mod:`rulfuse.degradation`);
* R1: the same extrapolation on the measured, causally smoothed HI (not deployable).

Their scores are on each fold's validation unit only: the test units stay sealed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from rulfuse import metrics as M
from rulfuse.core.folds import Fold
from rulfuse.core.normalization import Scaler, causal_baseline
from rulfuse.core.store import SegmentStore, image_visible
from rulfuse.degradation import DRAWS, DegradationPrior, predict_rul_many
from rulfuse.features.cache import ImageFeatures, SignalFeatures
from rulfuse.features.check import _recording_of_images, image_vectors
from rulfuse.features.probe import cross_fitted_readout, fit_ridge_cv


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


# ----------------------------------------------------------------------- B4, B5, R1

LAST_SEGMENTS = 3  # B4: the level descriptors are averaged over this many segments


@dataclass(frozen=True)
class PointPrediction:
    """Predictions of one unit's prediction points (``rows`` index the store's segments)."""

    rows: np.ndarray
    prediction: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


@dataclass(frozen=True)
class FoldResult:
    """A feature baseline's predictions per unit asked for, and what it fitted."""

    predictions: dict[str, PointPrediction]
    info: dict = field(default_factory=dict)


def _ordered_rows(store: SegmentStore, unit: str) -> np.ndarray:
    seg = store.segments
    rows = store.segment_rows(unit)
    return rows[np.lexsort((seg["segment"][rows], seg["recording"][rows]))]


def unit_level_features(
    store: SegmentStore, features: SignalFeatures, unit: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """B4 inputs of one unit's segments in time order (PROJECT_PLAN.md §8, P16): the segment
    rows, the absolute and the baseline-relative level descriptors (segments x channels*7),
    each averaged over the last 3 segments up to and including the segment (causal).

    A segment without a signal carries the last available descriptors forward (zeros before
    the first); the baseline is the mean of the unit's baseline segments seen so far."""
    seg = store.segments
    rows = _ordered_rows(store, unit)
    distinct = seg["distinct_segment"][rows].astype(np.int64)
    present = distinct >= 0
    n_cols = int(np.prod(features.descriptors.shape[1:]))
    raw = np.zeros((len(rows), n_cols))
    if present.any():
        ids = distinct[present]
        raw[present] = np.asarray(features.descriptors[ids], dtype=np.float64).reshape(len(ids), -1)
    raw = np.nan_to_num(raw)
    last = np.maximum.accumulate(np.where(present, np.arange(len(rows)), -1))
    carried = np.where((last >= 0)[:, None], raw[np.maximum(last, 0)], 0.0)
    in_baseline = seg["in_baseline"][rows].astype(bool)
    relative = carried - causal_baseline(carried, in_baseline, valid=present)

    def last_mean(x: np.ndarray) -> np.ndarray:
        total = np.cumsum(x, axis=0)
        total[LAST_SEGMENTS:] -= total[:-LAST_SEGMENTS].copy()
        count = np.minimum(np.arange(1, len(x) + 1), LAST_SEGMENTS)[:, None]
        return total / count

    return rows, last_mean(carried), last_mean(relative)


def _point_rows(store: SegmentStore, unit: str) -> np.ndarray:
    seg = store.segments
    return np.flatnonzero((seg["unit"] == unit) & seg["is_prediction_point"].astype(bool))


def b4_fold(
    store: SegmentStore,
    fold: Fold,
    signal_features: SignalFeatures,
    predict: Sequence[str] | None = None,
) -> FoldResult:
    """B4 on one fold (PROJECT_PLAN.md §8, P16): ridge (alpha by leave-one-unit-out over the
    training units) on [absolute descriptors, descriptors relative to the unit's causal
    baseline (both standardized on the training units), tau/H, elapsed/H] against the capped
    RUL, fitted on the training units' prediction points; predictions clipped to [0, H].
    ``predict`` names the units to predict (default: the fold's validation and test units)."""
    horizon, seg = store.settings.horizon_s, store.segments
    per_unit = {u: unit_level_features(store, signal_features, u) for u in store.unit_names}
    scaler_abs = Scaler.fit(np.concatenate([per_unit[u][1] for u in fold.train]))
    scaler_rel = Scaler.fit(np.concatenate([per_unit[u][2] for u in fold.train]))

    def design(unit: str) -> tuple[np.ndarray, np.ndarray]:
        rows, absolute, relative = per_unit[unit]
        keep = seg["is_prediction_point"][rows].astype(bool)
        usage = np.stack([seg["tau_s"][rows], seg["elapsed_s"][rows]], axis=1) / horizon
        x = np.concatenate(
            [scaler_abs.transform(absolute), scaler_rel.transform(relative), usage], axis=1
        )
        return rows[keep], x[keep]

    train = [design(u) for u in fold.train]
    x_train = np.concatenate([x for _, x in train])
    rows_train = np.concatenate([r for r, _ in train])
    y_train = seg["y_s"][rows_train].astype(float)
    groups = np.asarray(seg["unit"][rows_train]).astype(str)
    model = fit_ridge_cv(x_train, y_train, groups)
    out = {}
    for unit in predict if predict is not None else (fold.validation, fold.test):
        rows, x = design(unit)
        out[unit] = PointPrediction(rows, np.clip(model.predict(x), 0.0, horizon))
    info = {
        "alpha": model.alpha,
        "train_points": int(len(y_train)),
        "features": int(x_train.shape[1]),
        "scaler_mean_abs": scaler_abs.mean,
    }
    return FoldResult(out, info)


def _unit_curve(
    store: SegmentStore, unit: str, column: str = "hi"
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """A unit's smoothed HI against recording end time up to its failure recording, and the
    HI at failure; None for a unit that did not fail or has no usable HI."""
    rec = store.recordings
    failure_key = store.units[unit].get("failure_recording") or ""
    rows = store.recording_rows(unit)
    rows = rows[np.argsort(rec["index"][rows])]
    keys = [str(k) for k in rec["key"][rows]]
    if not failure_key or failure_key not in keys:
        return None
    rows = rows[: keys.index(failure_key) + 1]
    hi, end = rec[column][rows].astype(float), rec["end_s"][rows].astype(float)
    ok = np.isfinite(hi)
    if ok.sum() < 3 or not np.isfinite(hi[-1]):
        return None
    return end[ok], hi[ok], float(hi[-1])


def fit_prior(
    store: SegmentStore, train: Sequence[str], column: str = "hi"
) -> DegradationPrior:
    """The degradation prior of PROJECT_PLAN.md §8 on the training units' HI (``column``)."""
    curves = [c for c in (_unit_curve(store, u, column) for u in train) if c is not None]
    return DegradationPrior.fit([(t, h) for t, h, _ in curves], [f for _, _, f in curves])


def _extrapolate(
    store: SegmentStore,
    prior: DegradationPrior,
    unit: str,
    hist_t: np.ndarray,
    hist_hi: np.ndarray,
    draws: int,
    seed: int,
) -> PointPrediction:
    """Forecasts at the unit's prediction points from the history visible at each tau."""
    seg, horizon = store.segments, store.settings.horizon_s
    ok = np.isfinite(hist_hi) & np.isfinite(hist_t)
    order = np.argsort(hist_t[ok], kind="stable")
    hist_t, hist_hi = hist_t[ok][order], hist_hi[ok][order]
    rows = _point_rows(store, unit)
    tau = seg["tau_s"][rows].astype(float)
    n_visible = np.array([int(image_visible(hist_t, t).sum()) for t in tau])
    forecasts = predict_rul_many(prior, hist_t, hist_hi, tau, n_visible, horizon, draws, seed)
    mean, lower, upper = (np.array([f[i] for f in forecasts]) for i in range(3))
    return PointPrediction(rows, mean, lower, upper)


def b5_fold(
    store: SegmentStore,
    fold: Fold,
    image_features: ImageFeatures,
    image_kind: str,
    predict: Sequence[str] | None = None,
    draws: int = DRAWS,
    seed: int = 0,
) -> FoldResult:
    """B5 on one fold (PROJECT_PLAN.md §8, §6.5, P16): HI readouts of the images of
    ``image_kind`` (cross-fitted ridge on the training units' recording-level HI, so no
    readout used its own unit's labels), read at the images visible at each tau (times =
    availability), then the degradation extrapolation. The prior comes from the training
    units' smoothed HI; the likelihood noise is the fit residual sd combined with the
    cross-fitted readout residual sd on the training units."""
    rec = store.recordings
    image_rows, x = image_vectors(store, image_features, image_kind)
    rec_rows = _recording_of_images(store, image_rows)
    keep = rec_rows >= 0
    image_rows, x, rec_rows = image_rows[keep], x[keep], rec_rows[keep]
    units = store.images["unit"][image_rows].astype(str)
    target = rec["hi"][rec_rows].astype(float)
    readout = cross_fitted_readout(x, target, units, fold.train)
    train = np.isin(units, fold.train) & np.isfinite(target) & np.isfinite(readout)
    readout_sd = float(np.sqrt(np.mean((readout[train] - target[train]) ** 2)))
    prior = fit_prior(store, fold.train)
    fit_sd = prior.noise_sd
    prior = prior.with_noise(float(np.hypot(fit_sd, readout_sd)))
    out = {}
    for unit in predict if predict is not None else (fold.validation, fold.test):
        mine = units == unit
        out[unit] = _extrapolate(
            store,
            prior,
            unit,
            store.images["available_s"][image_rows[mine]].astype(float),
            readout[mine],
            draws,
            seed,
        )
    info = {
        "threshold": prior.threshold,
        "noise_sd": prior.noise_sd,
        "fit_noise_sd": fit_sd,
        "readout_sd": readout_sd,
        "prior_mean": prior.mean,
        "images": int(len(image_rows)),
    }
    return FoldResult(out, info)


def r1_fold(
    store: SegmentStore,
    fold: Fold,
    predict: Sequence[str] | None = None,
    draws: int = DRAWS,
    seed: int = 0,
    prior_column: str = "hi",
) -> FoldResult:
    """R1 on one fold (PROJECT_PLAN.md §8, P16; not deployable): the degradation
    extrapolation on the measured, causally smoothed HI (``hi_causal``) of the recordings
    that ended before tau, i.e. what a perfect image reader would give. The prior is fitted
    on the training units' ``prior_column`` (default: the smoothed HI, as for B5; the causal
    HI would match the history's measurement process)."""
    rec = store.recordings
    prior = fit_prior(store, fold.train, prior_column)
    out = {}
    for unit in predict if predict is not None else (fold.validation, fold.test):
        rows = store.recording_rows(unit)
        out[unit] = _extrapolate(
            store,
            prior,
            unit,
            rec["end_s"][rows].astype(float),
            rec["hi_causal"][rows].astype(float),
            draws,
            seed,
        )
    info = {"threshold": prior.threshold, "noise_sd": prior.noise_sd, "prior_mean": prior.mean}
    return FoldResult(out, info)


def _score_row(store: SegmentStore, unit: str, name: str, pred: PointPrediction) -> dict:
    seg = store.segments
    window = seg["window"][pred.rows].astype(bool)
    y = seg["rul_s"][pred.rows][window].astype(float)
    p = pred.prediction[window]
    row = {
        "fold_validation": unit,
        "baseline": name,
        "points": int(window.sum()),
        "mae": M.mae(y, p),
        "rmse": M.rmse(y, p),
        "bias": M.bias(y, p),
    }
    if pred.lower is not None and pred.upper is not None:
        row["coverage90"] = M.coverage(y, pred.lower[window], pred.upper[window])
        row["winkler90"] = M.winkler(y, pred.lower[window], pred.upper[window], 0.1)
    return row


def score_feature_baselines(
    store: SegmentStore,
    folds: Sequence[Fold],
    signal_features: SignalFeatures,
    image_features: ImageFeatures,
    image_kind: str,
    draws: int = DRAWS,
    seed: int = 0,
) -> dict:
    """B4, B5 and R1 (and B2 for reference) on each fold's VALIDATION unit, inside the window
    (PROJECT_PLAN.md §8, P16). Test units are never predicted or scored here: their scores
    stay sealed until the final evaluation. Per-fold rows and fold means."""
    horizon, seg = store.settings.horizon_s, store.segments
    rows: list[dict] = []
    fits: list[dict] = []
    for fold in folds:
        val = fold.validation
        results = {
            "B4": b4_fold(store, fold, signal_features, predict=(val,)),
            "B5": b5_fold(store, fold, image_features, image_kind, (val,), draws, seed),
            "R1": r1_fold(store, fold, (val,), draws, seed),
        }
        for name, result in results.items():
            rows.append(_score_row(store, val, name, result.predictions[val]))
            fits.append(
                {
                    "fold_validation": val,
                    "baseline": name,
                    **{
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in result.info.items()
                    },
                }
            )
        failures = np.array(
            [store.units[u]["failure_s"] for u in fold.train if store.units[u]["failure_s"]],
            dtype=float,
        )
        point = _point_rows(store, val)
        usage = PointPrediction(point, b2(failures, seg["tau_s"][point].astype(float), horizon))
        rows.append(_score_row(store, val, "B2", usage))
    summary: dict[str, dict[str, float]] = {}
    for name in ("B2", "B4", "B5", "R1"):
        mine = [row for row in rows if row["baseline"] == name]
        summary[name] = {
            metric: float(np.mean([row[metric] for row in mine]))
            for metric in ("mae", "rmse", "bias", "coverage90", "winkler90")
            if metric in mine[0]
        }
    return {
        "horizon_s": horizon,
        "scored_on": "validation units (test units sealed)",
        "image_kind": image_kind,
        "rows": rows,
        "fit": fits,
        "summary": summary,
    }
