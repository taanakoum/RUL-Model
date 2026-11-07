"""D23: the two inference modes (rulfuse/inference.py) and their use on B6's fit tables."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rulfuse.inference import CONTEXT_TIME_S, apply_mode, context_filter

H = 1500.0


def _ramp(failure_s: float, n: int, step: float = 1.0):
    tau = np.arange(n) * step
    return tau, np.minimum(failure_s - tau, H).clip(0.0)


def test_an_exact_trajectory_is_left_unchanged_cap_included():
    tau, y = _ramp(2500.0, 2500)  # 1,000 s at the cap, then 1,500 s of falling RUL
    np.testing.assert_allclose(context_filter(tau, y, H), y, atol=1e-9)
    gaps = np.cumsum(np.random.default_rng(0).integers(1, 4, 1200)).astype(float)
    exact = np.minimum(2000.0 - gaps, H).clip(0.0)  # uneven steps: ageing uses Δτ
    np.testing.assert_allclose(context_filter(gaps, exact, H), exact, atol=1e-9)


def test_the_context_mode_is_causal():
    rng = np.random.default_rng(1)
    tau, y = _ramp(1800.0, 1500)
    raw = y + rng.normal(0, 50, len(y))
    changed = raw.copy()
    changed[700:] = rng.uniform(0, H, len(y) - 700)  # rewrite the future
    a, b = context_filter(tau, raw, H), context_filter(tau, changed, H)
    np.testing.assert_array_equal(a[:700], b[:700])
    assert not np.allclose(a[700:], b[700:])


def test_it_recovers_from_an_early_prediction_that_was_too_low():
    tau, y = _ramp(1400.0, 1200)  # below the cap from the start
    raw = y.copy()
    raw[:200] -= 300.0  # the first 200 s are 300 s too low, then the model is right
    out = context_filter(tau, raw, H)
    error = y - out
    assert np.any(np.diff(out[199:260]) > 0)  # the output rises back: no lock-in
    later = 200 + int(3 * CONTEXT_TIME_S)  # the leftover error falls as exp(−elapsed/T)
    assert error[later] == pytest.approx(300.0 * np.exp(-(later - 199) / CONTEXT_TIME_S), rel=0.02)
    assert error[-1] == pytest.approx(300.0 * np.exp(-1000 / CONTEXT_TIME_S), rel=0.02)  # 0.07 s
    running_min = np.minimum.accumulate(raw)  # a "never rise" rule waits for the truth to fall
    t = 199 + int(CONTEXT_TIME_S)
    assert y[t] - running_min[t] == pytest.approx(300.0 - CONTEXT_TIME_S)  # 1 s per s only
    assert error[t] < 0.7 * (y[t] - running_min[t])


def test_it_removes_jitter_and_does_not_drift_at_the_cap():
    rng = np.random.default_rng(2)
    tau, y = _ramp(3000.0, 3000)
    raw = np.clip(y + rng.normal(0, 60, len(y)), 0, H)
    out = context_filter(tau, raw, H)
    window = y < H
    assert np.mean(np.abs(out[window] - y[window])) < np.mean(np.abs(raw[window] - y[window])) / 3
    capped = slice(200, 1400)  # true RUL > H here
    assert np.mean(out[capped]) > np.mean(raw[capped]) - 30.0  # no slide away from the cap


def test_modes_and_input_checks():
    tau, y = _ramp(1000.0, 50)
    np.testing.assert_array_equal(apply_mode("current", tau, y + 2000.0, H), np.full(50, H))
    np.testing.assert_allclose(apply_mode("context", tau, y, H), y)
    with pytest.raises(ValueError, match="unknown inference mode"):
        apply_mode("future", tau, y, H)
    with pytest.raises(ValueError, match="time order"):
        context_filter(tau[::-1], y, H)
    assert len(context_filter(np.zeros(0), np.zeros(0), H)) == 0


def test_b6_fit_tables_are_filtered_per_fit_unit_and_role():
    from rulfuse.omnifuser_rul.aggregate import with_mode

    rng = np.random.default_rng(3)
    tau_by_row = np.concatenate([np.arange(300.0), np.arange(200.0)])  # unit A rows, then B
    store: Any = SimpleNamespace(
        segments={"tau_s": tau_by_row}, settings=SimpleNamespace(horizon_s=H)
    )
    rows, unit, role, fold, seed = [], [], [], [], []
    for f, s, u, r, members in (
        (1, 0, "A", "test", np.arange(300)),
        (1, 1, "A", "test", np.arange(300)),
        (1, 0, "B", "validation", np.arange(300, 500)),
        (2, 0, "B", "test", np.arange(300, 500)),
    ):
        rows.append(members)
        unit += [u] * len(members)
        role += [r] * len(members)
        fold += [f] * len(members)
        seed += [s] * len(members)
    predictions = {
        "row": np.concatenate(rows),
        "unit": np.array(unit),
        "role": np.array(role),
        "fold": np.array(fold),
        "seed": np.array(seed),
        "prediction_s": rng.uniform(0, H, sum(len(m) for m in rows)),
    }
    shuffle = rng.permutation(len(predictions["row"]))
    predictions = {k: v[shuffle] for k, v in predictions.items()}
    out = with_mode(store, predictions, "context")
    assert with_mode(store, predictions, "current") is predictions
    for key in {(f, s, u, r) for f, s, u, r in zip(fold, seed, unit, role, strict=True)}:
        f, s, u, r = key
        mine = (
            (predictions["fold"] == f)
            & (predictions["seed"] == s)
            & (predictions["unit"] == u)
            & (predictions["role"] == r)
        )
        order = np.argsort(tau_by_row[predictions["row"][mine]])
        expected = context_filter(
            tau_by_row[predictions["row"][mine]][order], predictions["prediction_s"][mine][order], H
        )
        np.testing.assert_allclose(out["prediction_s"][mine][order], expected)
    np.testing.assert_array_equal(out["row"], predictions["row"])
