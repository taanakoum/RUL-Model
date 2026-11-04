"""Model channels and the 7 level descriptors (PROJECT_PLAN.md §5.4, §6.2; part P8)."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.adapters.nonastreda.dataset import CHANNELS as FORCE
from rulfuse.adapters.toy import CHANNELS as ACCELERATION
from rulfuse.core.channels import (
    DESCRIPTOR_NAMES,
    descriptors,
    expand,
    head_mean,
    level_descriptors,
    magnitude_groups,
    model_channels,
    slow_summary,
)
from rulfuse.core.schema import ChannelSpec


def test_magnitude_only_for_complete_three_axis_groups():
    names = [c.name for c in model_channels(FORCE)]
    assert names == ["Fx", "Fy", "Fz", "|force|"]
    assert model_channels(ACCELERATION) == ACCELERATION  # two axes: no magnitude
    mixed = (*FORCE, ChannelSpec("mic", "sound"))
    assert magnitude_groups(mixed) == [("force", (0, 1, 2))]
    out = expand(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]]), FORCE)
    np.testing.assert_allclose(out, [[3, 4, 0, 5], [0, 0, -2, 2]])


def test_vectorized_descriptors_equal_project14_functions():
    rng = np.random.default_rng(0)
    rate = 1000.0
    segment = rng.normal(size=(2000, 3)) * [1.0, 5.0, 0.1] + [0.0, 100.0, -3.0]
    head = segment[:100].mean(axis=0)
    out = descriptors(segment, rate, head)
    assert out.shape == (3, len(DESCRIPTOR_NAMES))
    for c in range(3):
        expected = np.concatenate(
            [level_descriptors(segment[:, c]), slow_summary(segment[:, c], rate)]
        )
        np.testing.assert_allclose(out[c, :6], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, 6], segment.mean(axis=0) - head, rtol=1e-5)


def test_descriptors_on_signals_with_known_values():
    rate = 1000.0
    t = np.arange(10_000) / rate
    sine = 2.0 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)  # RMS 2, peak 2√2, kurtosis 1.5
    ramp = 7.0 + 0.5 * t  # slope 0.5 per second
    x = np.stack([sine, ramp], axis=1)
    out = dict(zip(DESCRIPTOR_NAMES, descriptors(x, rate, head_mean(x, rate)).T, strict=True))
    assert out["log10_rms"][0] == pytest.approx(np.log10(2.0), abs=1e-3)
    assert out["crest"][0] == pytest.approx(np.sqrt(2), abs=1e-3)
    assert out["kurtosis"][0] == pytest.approx(1.5, abs=1e-3)
    assert out["slope"][1] == pytest.approx(0.5, abs=1e-6)
    assert out["mean"][1] == pytest.approx(7.0 + 0.5 * 4.9995, abs=1e-4)
    assert out["drift"][1] == pytest.approx(0.5 * (4.9995 - 0.4995), abs=1e-4)  # vs first 1 s


def test_level_survives_where_a_normalized_encoder_cannot_see_it():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 1))
    a, b = descriptors(x, 1000.0, np.zeros(1)), descriptors(2 * x, 1000.0, np.zeros(1))
    assert b[0, 0] - a[0, 0] == pytest.approx(np.log10(2), abs=1e-6)  # log RMS moves by log 2
    assert b[0, 2] == pytest.approx(a[0, 2], rel=1e-6)  # kurtosis is scale-free


def test_head_mean_is_causal():
    rng = np.random.default_rng(2)
    rec = rng.normal(size=(5000, 2))
    changed = rec.copy()
    changed[1000:] = 1e6  # later samples never change the drift reference
    np.testing.assert_array_equal(head_mean(rec, 1000.0), head_mean(changed, 1000.0))
