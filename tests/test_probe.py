"""P9 Probes, cross-fitted readouts and the leakage-probe PCA (PROJECT_PLAN.md §6.2)."""

from __future__ import annotations

import numpy as np

from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.normalization import PCA, Scaler
from rulfuse.features.check import loo_pcs
from rulfuse.features.probe import cross_fitted_readout, fit_ridge, fit_ridge_cv, probe_folds

UNITS = [f"U{i}" for i in range(6)]


def _data(seed: int = 0, n: int = 40, d: int = 5, noise: float = 0.1):
    rng = np.random.default_rng(seed)
    groups = np.repeat(UNITS, n)
    x = rng.normal(size=(len(groups), d))
    coef = np.arange(1, d + 1, dtype=float)
    y = x @ coef + 3.0 + rng.normal(0, noise, len(groups))
    return x, y, groups, coef


def test_ridge_recovers_known_coefficients():
    x, y, groups, coef = _data()
    model = fit_ridge(x, y, alpha=1e-6)
    np.testing.assert_allclose(model.coef / model.scaler.scale, coef, rtol=0.02)
    cv = fit_ridge_cv(x, y, groups)
    assert np.mean(np.abs(cv.predict(x) - y)) < 0.2


def test_cross_fitted_readout_never_uses_its_own_units_labels():
    x, y, groups, _ = _data(1)
    train = UNITS[:5]
    base = cross_fitted_readout(x, y, groups, train)
    poisoned = y.copy()
    poisoned[groups == "U0"] += 1e4  # U0's own labels change wildly
    again = cross_fitted_readout(x, poisoned, groups, train)
    np.testing.assert_allclose(again[groups == "U0"], base[groups == "U0"])
    outside = y.copy()
    outside[groups == "U5"] = -1e4  # labels of a unit outside training are never read
    np.testing.assert_allclose(cross_fitted_readout(x, outside, groups, train), base)


def test_probe_beats_the_mean_and_the_shuffled_control_collapses():
    x, y, groups, _ = _data(2)
    folds = leave_one_unit_out(UNITS)
    real = probe_folds(x, y, groups, folds)
    assert real["beats_mean"] and real["wins"] == len(folds)
    shuffled = probe_folds(x, y, groups, folds, shuffle_seed=0)
    assert shuffled["mae"] > 0.8 * shuffled["mae_mean_predictor"]
    noise = np.random.default_rng(3).normal(size=x.shape)
    assert not probe_folds(noise, y, groups, folds)["beats_mean"]


def test_leakage_pca_is_fitted_without_the_projected_unit():
    x, _, groups, _ = _data(4)
    pcs = loo_pcs(x, groups, k=2)
    own = groups == "U2"
    scaler = Scaler.fit(x[~own])
    pca = PCA.fit(scaler.transform(x[~own]), 2)
    np.testing.assert_allclose(pcs[own], pca.transform(scaler.transform(x[own])))
    shifted = x.copy()
    shifted[own] *= 50  # U2's own data never enters the basis it is projected on
    np.testing.assert_allclose(
        loo_pcs(shifted, groups, 2)[own], pca.transform(scaler.transform(shifted[own]))
    )
