"""P7 Normalization: training-only fitting and the causal baseline (PROJECT_PLAN.md §5.9)."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.data.folds import leave_one_tool_out_folds
from rulfuse.data.normalization import (
    PCA,
    BlockTransform,
    Scaler,
    baseline_rows,
    causal_baseline,
    relative_to_baseline,
    relative_to_first,
    train_rows,
)

TOOLS = [f"T{i}" for i in range(1, 11)]


def _features(seed: int = 0):
    rng = np.random.default_rng(seed)
    tools = np.repeat(TOOLS, 30)
    x = rng.normal(size=(len(tools), 6)) @ rng.normal(size=(6, 6))
    return tools, x


def test_poison_validation_and_test_rows_do_not_move_the_fit():
    tools, x = _features()
    fold = next(iter(leave_one_tool_out_folds(TOOLS)))
    mask = train_rows(tools, fold)
    assert set(tools[mask]) == set(fold.train_tools)
    poisoned = x.copy()
    poisoned[~mask] = 1e6
    a, b = Scaler.fit(x[mask]), Scaler.fit(poisoned[mask])
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.scale, b.scale)
    pa_, pb = PCA.fit(a.transform(x[mask]), 3), PCA.fit(b.transform(poisoned[mask]), 3)
    np.testing.assert_allclose(pa_.components, pb.components)
    blocks = {"a": [0, 1, 2], "b": [3, 4, 5]}
    ta, tb = BlockTransform.fit(x[mask], blocks, 2), BlockTransform.fit(poisoned[mask], blocks, 2)
    np.testing.assert_allclose(ta.transform(x), tb.transform(x))


def test_scaler_and_pca_properties():
    _, x = _features(1)
    scaled = Scaler.fit(x).transform(x)
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-12)
    constant = np.ones((5, 2))
    assert np.all(Scaler.fit(constant).scale == 1.0)  # floor: no division by zero
    pca = PCA.fit(scaled, 4)
    np.testing.assert_allclose(pca.components @ pca.components.T, np.eye(4), atol=1e-10)
    z = pca.transform(scaled)
    assert np.all(np.diff(z.var(axis=0)) <= 1e-12)  # variance in decreasing order
    again = PCA.fit(scaled, 4)
    np.testing.assert_array_equal(pca.components, again.components)  # deterministic signs
    full = PCA.fit(scaled, 6)
    np.testing.assert_allclose(full.transform(scaled) @ full.components + full.mean, scaled)


def test_block_transform_keeps_channels_separate():
    _, x = _features(2)
    blocks = {"Fx": [0, 1], "Fy": [2, 3], "Fz": [4, 5]}
    transform = BlockTransform.fit(x, blocks, 1)
    slices = transform.output_slices()
    assert slices == {"Fx": slice(0, 1), "Fy": slice(1, 2), "Fz": slice(2, 3)}
    changed = x.copy()
    changed[:, [0, 1]] += 100.0  # only the Fx block's inputs change
    a, b = transform.transform(x), transform.transform(changed)
    np.testing.assert_allclose(a[:, 1:], b[:, 1:])
    assert not np.allclose(a[:, 0], b[:, 0])


def test_causal_baseline_is_a_running_mean_of_the_past():
    values = np.arange(10, dtype=float)[:, None] * np.array([[1.0, 10.0]])
    in_base = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    base = causal_baseline(values, in_base)
    np.testing.assert_allclose(base[0], values[0])
    np.testing.assert_allclose(base[2], values[:3].mean(axis=0))
    np.testing.assert_allclose(base[4], values[:3].mean(axis=0))  # outside the baseline: frozen
    np.testing.assert_allclose(base[9], values[[0, 1, 2, 5]].mean(axis=0))
    assert np.allclose(relative_to_baseline(values, in_base)[0], 0)
    future = values.copy()
    future[6:] = 1e6  # the future never changes the past
    np.testing.assert_allclose(causal_baseline(future, in_base)[:6], base[:6])
    valid = np.ones(10, dtype=bool)
    valid[1] = False  # a force-missing token is not averaged in
    np.testing.assert_allclose(
        causal_baseline(values, in_base, valid)[2], values[[0, 2]].mean(axis=0)
    )


def test_rows_before_any_baseline_get_zero():
    values = np.ones((3, 2))
    base = causal_baseline(values, np.array([False, True, True]))
    np.testing.assert_allclose(base[0], 0.0)


def test_relative_to_first_and_baseline_rows():
    photos = np.array([[2.0], [3.0], [5.0]])
    np.testing.assert_allclose(relative_to_first(photos), [[0.0], [1.0], [3.0]])
    np.testing.assert_allclose(
        relative_to_first(photos, np.array([False, True, True])), [[0.0], [0.0], [2.0]]
    )
    rows = baseline_rows(
        np.array([1, 1, 2, 3, 9, 10]), recorded_positions=[9, 1, 2, 3, 10], passes=3
    )
    assert rows.tolist() == [True, True, True, True, False, False]


def test_pca_rejects_empty_input():
    with pytest.raises(ValueError):
        PCA.fit(np.zeros((0, 3)), 2)
