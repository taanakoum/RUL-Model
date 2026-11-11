"""P16 (part) Degradation extrapolation (B5, R1) and the B4 ridge baseline
(PROJECT_PLAN.md §8, §15.1)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rulfuse.adapters import toy
from rulfuse.baselines import (
    _point_rows,
    b4_fold,
    r1_fold,
    score_feature_baselines,
    unit_level_features,
)
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.degradation import (
    DegradationPrior,
    _forecast,
    fit_power_law,
    power_law,
    predict_rul,
)
from rulfuse.features.cache import extract_image_features, extract_signal_features
from rulfuse.features.encoders import PixelStatsEncoder, ProjectionEncoder

HORIZON = 600.0
THETA = 100.0
REAL_STORE = Path(r"D:\RULModel\interim\segments\nonastreda\3e357b5a23ff")


def _unit(life: float, gamma: float, n: int = 40, noise: float = 0.0, seed: int = 0):
    """(τ, HI) of a power-law unit that reaches THETA at ``life``."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(life / n, life, n)
    hi = 20.0 + (THETA - 20.0) * (tau / life) ** gamma + rng.normal(0.0, noise, n)
    return tau, hi


def _prior(noise: float = 0.5) -> DegradationPrior:
    lives = [900.0, 1000.0, 1100.0, 950.0, 1050.0, 1000.0]
    gammas = [1.6, 1.5, 1.7, 1.55, 1.65, 1.6]
    units = [
        _unit(life, g, noise=noise, seed=i)
        for i, (life, g) in enumerate(zip(lives, gammas, strict=True))
    ]
    return DegradationPrior.fit(units, [THETA] * len(units))


# ------------------------------------------------------------------- power-law fit


def test_fit_power_law_recovers_known_parameters():
    tau = np.linspace(20.0, 1000.0, 60)
    truth = np.array([20.0, np.log(80.0), np.log(1.8)])
    hi = power_law(tau, truth, time_scale=1000.0)
    params, sd = fit_power_law(tau, hi, time_scale=1000.0)
    np.testing.assert_allclose(params, truth, atol=1e-3)
    assert sd < 1e-6
    noisy = hi + np.random.default_rng(0).normal(0.0, 1.0, len(hi))
    params, sd = fit_power_law(tau, noisy, time_scale=1000.0)
    assert np.exp(params[2]) == pytest.approx(1.8, abs=0.5)
    assert sd == pytest.approx(1.0, abs=0.35)


def test_fit_power_law_keeps_gamma_at_least_one_and_beta_positive():
    tau = np.linspace(20.0, 1000.0, 40)
    concave = 20.0 + 80.0 * np.sqrt(tau / 1000.0)  # the true exponent is 0.5
    params, _ = fit_power_law(tau, concave, time_scale=1000.0)
    assert np.exp(params[2]) >= 1.0 - 1e-9 and np.isfinite(params[1])


def test_prior_collects_the_units():
    prior = _prior()
    assert prior.n_units == 6 and prior.threshold == THETA
    assert prior.cov.shape == (3, 3) and np.all(np.linalg.eigvalsh(prior.cov) > 0)
    assert 0.3 < prior.noise_sd < 1.0  # pooled residual sd of noise 0.5


# ------------------------------------------------------------------- extrapolation


def test_predict_rul_recovers_the_crossing_mid_life():
    life = 960.0
    tau, hi = _unit(life, 1.6, n=48, noise=0.5, seed=11)
    now = 480.0
    seen = tau <= now
    forecast = predict_rul(_prior(), tau[seen], hi[seen], now, HORIZON, seed=1)
    assert forecast.mean == pytest.approx(life - now, abs=0.05 * HORIZON)
    assert forecast.lower <= forecast.mean <= forecast.upper


def test_noiseless_history_is_near_exact():
    life = 1000.0
    tau, hi = _unit(life, 1.6, n=50)
    for now in (600.0, 800.0):
        seen = tau <= now
        forecast = predict_rul(_prior(noise=0.2), tau[seen], hi[seen], now, HORIZON, seed=2)
        assert forecast.mean == pytest.approx(life - now, abs=0.02 * HORIZON)


def test_forecast_weights_normalise_and_the_prior_alone_is_uniform():
    prior = _prior()
    alone = predict_rul(prior, np.array([]), np.array([]), 300.0, HORIZON, draws=2000, seed=3)
    assert alone.ess == pytest.approx(2000.0)  # no history: equal weights
    tau, hi = _unit(1000.0, 1.6, n=40, noise=0.5)
    informed = predict_rul(prior, tau[:20], hi[:20], tau[19], HORIZON, draws=2000, seed=3)
    assert 1.0 <= informed.ess < 2000.0
    rul = np.linspace(0.0, HORIZON, 200)
    order = np.arange(200)
    log_w = np.random.default_rng(0).normal(size=200)
    base = _forecast(rul, order, log_w)
    shifted = _forecast(rul, order, log_w + 1234.5)  # unnormalised weights, same forecast
    np.testing.assert_allclose(base[:3], shifted[:3])


def test_forecast_is_clipped_to_zero_and_the_horizon():
    prior = _prior()
    tau, _ = _unit(1000.0, 1.6, n=40)
    over = predict_rul(prior, tau[:30], np.full(30, 1.5 * THETA), tau[29], HORIZON, seed=4)
    assert over.mean == 0.0 and over.upper == 0.0  # history far above θ: crossed already
    slow = DegradationPrior(
        np.array([20.0, np.log(1e-3), 0.0]), 1e-12 * np.eye(3), 1e6, 1.0, 1000.0, 3
    )
    never = predict_rul(slow, np.array([]), np.array([]), 100.0, HORIZON, draws=500, seed=0)
    assert never.mean == HORIZON and never.lower == HORIZON
    for now in (0.0, 500.0, 5000.0):
        f = predict_rul(prior, np.array([]), np.array([]), now, HORIZON, draws=500, seed=0)
        assert 0.0 <= f.lower <= f.mean <= f.upper <= HORIZON


# ------------------------------------------------------------------- B4 on a toy store


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy_b4")
    ds = toy.build(root / "images")
    store = SegmentStore.open(build_segment_store(ds, root / "store"))
    signal = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=root / "f", log=lambda *_: None
    )
    image = extract_image_features(store, PixelStatsEncoder(), root=root / "f", log=lambda *_: None)
    return store, signal, image


def _shifted(signal, rows: np.ndarray, store: SegmentStore, factor: float):
    """A copy of the descriptor cache with the given segment rows' descriptors rescaled."""
    descriptors = np.array(signal.descriptors, dtype=np.float64)
    ids = store.segments["distinct_segment"][rows]
    descriptors[ids] = descriptors[ids] * factor + 7.0
    return SimpleNamespace(descriptors=descriptors)


def test_b4_shapes_and_range(toy_run):
    store, signal, _ = toy_run
    fold = leave_one_unit_out(store.unit_names)[0]
    result = b4_fold(store, fold, signal)
    assert set(result.predictions) == {fold.validation, fold.test}
    for unit, prediction in result.predictions.items():
        rows = _point_rows(store, unit)
        np.testing.assert_array_equal(prediction.rows, rows)
        assert prediction.prediction.shape == rows.shape
        assert prediction.prediction.min() >= 0.0 and prediction.prediction.max() <= HORIZON
    n_channels, n_desc = signal.descriptors.shape[1:]
    assert result.info["features"] == 2 * n_channels * n_desc + 2
    only = b4_fold(store, fold, signal, predict=(fold.validation,))
    assert set(only.predictions) == {fold.validation}
    np.testing.assert_allclose(
        only.predictions[fold.validation].prediction,
        result.predictions[fold.validation].prediction,
    )


def test_b4_is_causal(toy_run):
    store, signal, _ = toy_run
    fold = leave_one_unit_out(store.unit_names)[0]
    rows, *_ = unit_level_features(store, signal, fold.validation)
    cut = len(rows) // 2
    changed = _shifted(signal, rows[cut:], store, 3.0)
    before = b4_fold(store, fold, signal, predict=(fold.validation,)).predictions[  # pyright: ignore[reportArgumentType]
        fold.validation
    ]
    after = b4_fold(store, fold, changed, predict=(fold.validation,)).predictions[  # pyright: ignore[reportArgumentType]
        fold.validation
    ]
    early = np.isin(before.rows, rows[:cut])
    assert early.any() and (~early).any()
    np.testing.assert_allclose(after.prediction[early], before.prediction[early])
    assert not np.allclose(after.prediction[~early], before.prediction[~early])


def test_b4_scaling_uses_only_the_training_units(toy_run):
    store, signal, _ = toy_run
    fold = leave_one_unit_out(store.unit_names)[0]
    base = b4_fold(store, fold, signal)
    expected = np.concatenate([unit_level_features(store, signal, u)[1] for u in fold.train]).mean(
        axis=0
    )
    np.testing.assert_allclose(base.info["scaler_mean_abs"], expected)
    for unit in (fold.validation, fold.test):  # wreck the held-out units' descriptors
        rows, *_ = unit_level_features(store, signal, unit)
        wrecked = _shifted(signal, rows, store, 50.0)
        other = b4_fold(store, fold, wrecked)  # pyright: ignore[reportArgumentType]
        np.testing.assert_allclose(other.info["scaler_mean_abs"], base.info["scaler_mean_abs"])
        assert other.info["alpha"] == base.info["alpha"]


def test_b4_carries_the_last_descriptors_forward_without_a_signal(toy_run):
    store, signal, _ = toy_run
    unit = store.unit_names[0]
    rows, full_abs, _ = unit_level_features(store, signal, unit)
    seg = {k: np.array(v) for k, v in store.segments.items()}
    seg["distinct_segment"][rows[[0, 6]]] = -1
    seg["in_baseline"][rows[[0, 6]]] = False
    holes = dataclasses.replace(store, segments=seg)
    _, absolute, relative = unit_level_features(holes, signal, unit)
    raw = np.asarray(signal.descriptors, dtype=np.float64).reshape(len(signal.descriptors), -1)
    d = store.segments["distinct_segment"][rows]
    np.testing.assert_allclose(absolute[0], 0.0)  # nothing available before it
    np.testing.assert_allclose(relative[0], 0.0)
    carried = raw[d[5]]  # segment 6 repeats segment 5
    np.testing.assert_allclose(absolute[6], (raw[d[4]] + raw[d[5]] + carried) / 3.0)
    np.testing.assert_allclose(absolute[10], full_abs[10])  # later windows are unaffected


def test_r1_runs_on_the_toy_store(toy_run):
    store, *_ = toy_run
    fold = leave_one_unit_out(store.unit_names)[0]
    result = r1_fold(store, fold, predict=(fold.validation,), draws=1000)
    prediction = result.predictions[fold.validation]
    seg = store.segments
    window = seg["window"][prediction.rows].astype(bool)
    error = np.abs(prediction.prediction[window] - seg["rul_s"][prediction.rows][window])
    assert error.mean() < HORIZON / 2  # the toy HI crosses its threshold at the failure
    assert prediction.lower is not None and prediction.upper is not None
    assert (prediction.lower <= prediction.upper).all()


def test_score_feature_baselines_scores_validation_units_only(toy_run):
    store, signal, image = toy_run
    folds = leave_one_unit_out(store.unit_names)[:2]
    result = score_feature_baselines(store, folds, signal, image, "surface", draws=500)
    assert {r["baseline"] for r in result["rows"]} == {"B2", "B4", "B5", "R1"}
    assert {r["fold_validation"] for r in result["rows"]} == {f.validation for f in folds}
    assert set(result["summary"]) == {"B2", "B4", "B5", "R1"}
    assert all(r["mae"] >= 0.0 for r in result["rows"])


@pytest.mark.data
@pytest.mark.skipif(not REAL_STORE.is_dir(), reason="real segment store absent")
def test_real_store_first_fold_runs():
    from rulfuse.features.cache import ImageFeatures, SignalFeatures, find_features

    store = SegmentStore.open(REAL_STORE)
    signal = SignalFeatures.open(find_features(store, "signal", "mantis8m"))
    image = ImageFeatures.open(find_features(store, "image", "dinov2b"))
    folds = leave_one_unit_out(store.unit_names)[:1]
    result = score_feature_baselines(store, folds, signal, image, "tool", draws=500)
    assert {r["fold_validation"] for r in result["rows"]} == {folds[0].validation}
    assert all(np.isfinite(r["mae"]) for r in result["rows"])
