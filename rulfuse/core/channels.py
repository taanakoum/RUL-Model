"""Model channels and level descriptors (PROJECT_PLAN.md §5.4, §6.2).

**Channels.** Every measured channel is used. One derived channel is added by a generic
rule: the magnitude of each sensor's raw x, y, z group (here |F| = √(Fx² + Fy² + Fz²)).

**Level descriptors** (7 per channel and segment), what scale-normalized encoders cannot
see. The first six are vendored from Project 14 (``pdmfm/preprocess.py``:
``level_descriptors`` and ``slow_summary``, PREPROCESS_VERSION 2):

* log10 RMS and log10 peak of the AC part, Pearson kurtosis (normal = 3), crest factor;
* mean, least-squares slope (units per second);
* ``drift``: the segment mean minus the mean of the recording's first second, the drift a
  recording accumulates. It is causal: the head is known before the first segment ends.

They need no machine parameters, so they exist for any signal of any dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from rulfuse.core.schema import ChannelSpec

DESCRIPTOR_NAMES = ("log10_rms", "log10_peak", "kurtosis", "crest", "mean", "slope", "drift")
DESCRIPTORS_VERSION = 1
DRIFT_HEAD_SECONDS = 1.0
EPS = 1e-12


def magnitude_groups(channels: Sequence[ChannelSpec]) -> list[tuple[str, tuple[int, int, int]]]:
    """(sensor, column indices of x, y, z) of every sensor with three raw axes."""
    groups = []
    sensors = dict.fromkeys(c.sensor for c in channels)
    for sensor in sensors:
        axes = {
            c.axis: index
            for index, c in enumerate(channels)
            if c.sensor == sensor and c.derivation == "raw"
        }
        if all(axis in axes for axis in ("x", "y", "z")):
            groups.append((sensor, (axes["x"], axes["y"], axes["z"])))
    return groups


def model_channels(channels: Sequence[ChannelSpec]) -> tuple[ChannelSpec, ...]:
    """The measured channels, then one magnitude channel per 3-axis group."""
    derived = []
    for sensor, index in magnitude_groups(channels):
        unit = channels[index[0]].unit
        derived.append(ChannelSpec(f"|{sensor}|", sensor, "none", "magnitude", unit))
    return (*channels, *derived)


def expand(x: np.ndarray, channels: Sequence[ChannelSpec]) -> np.ndarray:
    """(samples, measured) -> (samples, model channels) float32."""
    x = np.asarray(x, dtype=np.float32)
    parts = [x]
    for _, index in magnitude_groups(channels):
        parts.append(np.sqrt(np.square(x[:, list(index)]).sum(axis=1, keepdims=True)))
    return np.concatenate(parts, axis=1)


def level_descriptors(window: np.ndarray) -> np.ndarray:
    """(4,) for one channel (vendored from Project 14, unchanged)."""
    x = np.asarray(window, dtype=np.float64)
    ac = x - x.mean()
    rms = float(np.sqrt(np.mean(ac**2)))
    peak = float(np.max(np.abs(ac)))
    kurt = float(stats.kurtosis(ac, fisher=False)) if rms > 0 else 0.0
    return np.array(
        [np.log10(rms + EPS), np.log10(peak + EPS), kurt, peak / (rms + EPS)], dtype=np.float32
    )


def slow_summary(window: np.ndarray, rate_hz: float) -> np.ndarray:
    """(2,) mean and least-squares slope per second (vendored from Project 14, unchanged)."""
    x = np.asarray(window, dtype=np.float64)
    t = np.arange(len(x)) / rate_hz
    slope = np.polyfit(t, x, 1)[0] if len(x) > 1 else 0.0
    return np.array([x.mean(), slope], dtype=np.float32)


def head_mean(recording: np.ndarray, rate_hz: float) -> np.ndarray:
    """Per channel, the mean over the recording's first second (the drift reference)."""
    head = np.asarray(recording)[: max(1, int(DRIFT_HEAD_SECONDS * rate_hz))]
    return head.mean(axis=0, dtype=np.float64)


def descriptors(segment: np.ndarray, rate_hz: float, head: np.ndarray) -> np.ndarray:
    """(channels, 7) for a (samples, channels) segment; vectorized over channels and equal
    to ``level_descriptors`` + ``slow_summary`` + drift channel by channel."""
    x = np.asarray(segment, dtype=np.float64)
    n = len(x)
    mean = x.mean(axis=0)
    ac = x - mean
    rms = np.sqrt(np.mean(ac**2, axis=0))
    peak = np.max(np.abs(ac), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        kurt = np.where(rms > 0, stats.kurtosis(ac, axis=0, fisher=False), 0.0)
    if n > 1:
        t = np.arange(n) / rate_hz
        tc = t - t.mean()
        slope = (tc @ ac) / float(tc @ tc)
    else:
        slope = np.zeros(x.shape[1])
    out = np.stack(
        [
            np.log10(rms + EPS),
            np.log10(peak + EPS),
            np.nan_to_num(kurt),
            peak / (rms + EPS),
            mean,
            slope,
            mean - np.asarray(head, dtype=np.float64),
        ],
        axis=1,
    )
    return out.astype(np.float32)
