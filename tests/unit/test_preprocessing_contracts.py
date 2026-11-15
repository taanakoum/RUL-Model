"""P2-P5/P8: independent arithmetic, randomized tiling, and EDA failure cases."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.adapters.nonastreda.failure import isotonic_causal, ttf_index
from rulfuse.adapters.nonastreda.trim import span_bounds_causal
from rulfuse.baselines import b2
from rulfuse.core.channels import descriptors, expand, model_channels
from rulfuse.core.images import normalize, standardize_luminance
from rulfuse.core.schema import ChannelSpec
from rulfuse.core.segments import segment_ends, segment_sample_bounds, subwindow_starts
from rulfuse.core.targets import prediction_mask, rul_targets
from rulfuse.features.cache import resample
from tests.helpers import tool_metas

pytestmark = [pytest.mark.unit, pytest.mark.requirements("P2", "P3", "P4", "P5", "P8", "P16")]


@pytest.mark.parametrize("seed", range(8))
def test_sample_tiling_has_no_gaps_overlap_or_zero_length_segments(seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):
        rate = int(rng.choice([32, 100, 1000, 2000]))
        length = float(rng.choice([0.5, 1, 2, 5, 10, 20]))
        n = int(rng.integers(1, rate * 45))
        bounds = segment_sample_bounds(n, rate, length, length / 2)
        assert bounds[0][0] == 0 and bounds[-1][1] == n
        assert all(a < b for a, b in bounds)
        assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
        assert sum(b - a for a, b in bounds) == n
        np.testing.assert_allclose([b / rate for _, b in bounds],
                                   segment_ends(n / rate, length, length / 2), atol=1 / rate)


@pytest.mark.parametrize("duration, expected", [
    (4.999, [4.999]), (10, [10]), (14.999, [14.999]),
    (15, [10, 15]), (20, [10, 20]), (24.999, [10, 24.999]),
])
def test_remainder_boundary_is_exact(duration, expected):
    np.testing.assert_allclose(segment_ends(duration), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("samples", [1, 511, 512, 513, 1023, 1024, 1537, 10000])
def test_encoder_subwindows_cover_every_sample(samples):
    length = 512
    starts = subwindow_starts(samples, length)
    covered = np.zeros(max(samples, length), bool)
    for start in starts:
        covered[start:start + length] = True
    assert covered[:samples].all()
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    assert starts[-1] == max(samples - length, 0)


@pytest.mark.parametrize("scale", [0.01, 1.0, 1000.0])
def test_targets_scale_with_physical_seconds_and_ignore_censoring(scale):
    tau = np.array([0, 10, 25, 40], float) * scale
    result = rul_targets(tau, 40 * scale, 20 * scale)
    np.testing.assert_allclose(result.rul_s, np.array([40, 30, 15, 0]) * scale)
    np.testing.assert_allclose(result.y_s, np.array([20, 20, 15, 0]) * scale)
    np.testing.assert_array_equal(result.window, [False, False, True, True])
    assert not prediction_mask(tau, None).any()


def test_b2_caps_each_surviving_life_before_averaging():
    # At t=10, residuals are [10, 990], so E[min(R,H)] = 55, not min(E[R],H) = 100.
    np.testing.assert_allclose(b2(np.array([20., 1000.]), np.array([10., 20., 1000.]), 100),
                               [55, 100, 0])


@pytest.mark.parametrize("labels, expected", [
    (["dulled", "sharp", "dulled", "dulled"], 0),
    (["sharp", "dulled", "used", "sharp", "sharp"], None),
    (["sharp", "dulled", "dulled"], None),
    (["sharp", "dulled", "dulled", "dulled"], 1),
])
def test_failure_confirmation_uses_next_three_recorded_labels(labels, expected):
    assert ttf_index("confirmed", tool_metas(labels)) == expected


def test_causal_hi_prefixes_do_not_change_when_future_wear_is_poisoned():
    wear = np.array([20., 25., 23., 35., 28., 50.])
    expected = isotonic_causal(wear)
    for cut in range(1, len(wear)):
        poisoned = wear.copy()
        poisoned[cut:] = 1e6
        np.testing.assert_array_equal(isotonic_causal(poisoned)[:cut], expected[:cut])


def test_low_force_cutting_ramp_is_retained_as_eda_requires():
    t = np.arange(15000) / 1000
    force = np.stack([13 + 4 * t + np.sin(2 * np.pi * 40 * t), np.zeros(len(t)),
                      np.zeros(len(t))], axis=1)
    start, end = span_bounds_causal(force)
    assert start == 0 and end == len(force)
    for cut in [6000, 9000, 12000]:
        assert span_bounds_causal(force[:cut])[0] == start


def test_channels_and_drift_have_analytic_physical_values():
    channels = tuple(ChannelSpec(axis, "force", axis) for axis in "xyz")
    t = np.arange(1000) / 100
    raw = np.stack([3 + 2 * t, np.full_like(t, 4), np.zeros_like(t)], axis=1)
    expanded = expand(raw, channels)
    assert len(model_channels(channels)) == 4
    np.testing.assert_allclose(expanded[:, -1], np.linalg.norm(raw, axis=1), rtol=1e-6)
    desc = descriptors(raw, 100, np.array([1., 4., 0.]))
    np.testing.assert_allclose(desc[:, 5], [2, 0, 0], atol=1e-6)
    np.testing.assert_allclose(desc[:, 6], raw.mean(0) - [1, 4, 0], atol=1e-6)
    assert np.isfinite(desc).all()


@pytest.mark.parametrize("source,destination", [(64, 32), (1000, 1000), (2000, 1000)])
def test_resampling_preserves_duration_and_channel_identity(source, destination):
    x = np.tile([1., 2., -3.], (source * 2, 1)).astype(np.float32)
    y = resample(x, source, destination)
    assert y.shape == (destination * 2, 3) and y.dtype == np.float32
    np.testing.assert_allclose(y[20:-20], np.tile([1, 2, -3], (len(y) - 40, 1)), atol=0.005)


def test_workpiece_standardization_removes_affine_exposure_without_clipping():
    grey = np.tile(np.linspace(.35, .65, 20), (20, 1))
    rgb = np.repeat(grey[..., None], 3, axis=2)
    np.testing.assert_allclose(standardize_luminance(rgb),
                               standardize_luminance(.6 * rgb + .15), atol=1e-6)
    prepared = normalize(np.full((8, 12, 3), .5, np.float32))
    assert prepared.shape == (3, 8, 12) and np.isfinite(prepared).all()
