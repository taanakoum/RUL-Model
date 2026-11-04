"""The handcrafted feature set H (PROJECT_PLAN.md §6.2): baseline B4 and diagnostics only.

H = the 7 level descriptors (:mod:`rulfuse.core.channels`) + these 11 spectral features
per channel, all defined relative to the model rate, so no machine parameter is needed:

* log10 energy in 8 octave bands below the Nyquist frequency N: [N/2, N], [N/4, N/2], …,
  [N/256, N/128];
* the spectral centroid as a fraction of N, and the normalized spectral entropy;
* log10 RMS of the first difference (high-frequency activity).

The Nonastreda-specific features of the first plan (the 66.7 Hz spindle line and the
period-15 synchronous average) are dropped (D17, D18).
"""

from __future__ import annotations

import numpy as np

SPECTRAL_NAMES = (
    *(f"log10_band{k}" for k in range(1, 9)),
    "centroid",
    "entropy",
    "log10_rms_diff",
)
EPS = 1e-12


def spectral_features(segment: np.ndarray) -> np.ndarray:
    """(channels, 11) for a (samples, channels) segment at the model rate."""
    x = np.asarray(segment, dtype=np.float64)
    x = x - x.mean(axis=0)
    n = len(x)
    power = np.abs(np.fft.rfft(x * np.hanning(n)[:, None], axis=0)) ** 2  # (bins, C)
    freq = np.linspace(0.0, 1.0, power.shape[0])  # fraction of Nyquist
    bands = []
    for k in range(1, 9):
        lo, hi = 2.0**-k, 2.0 ** -(k - 1)
        mask = (freq >= lo) & (freq < hi) if k > 1 else (freq >= lo)
        bands.append(np.log10(power[mask].sum(axis=0) + EPS))
    total = power.sum(axis=0) + EPS
    centroid = (freq[:, None] * power).sum(axis=0) / total
    p = power / total
    entropy = -(p * np.log(p + EPS)).sum(axis=0) / np.log(power.shape[0])
    diff = np.diff(x, axis=0) if n > 1 else np.zeros((1, x.shape[1]))
    rms_diff = np.log10(np.sqrt(np.mean(diff**2, axis=0)) + EPS)
    return np.stack([*bands, centroid, entropy, rms_diff], axis=1).astype(np.float32)


def mean_rms_names(channel_names: list[str]) -> tuple[str, ...]:
    return tuple(f"{stat}_{name}" for name in channel_names for stat in ("mean", "rms"))


def segment_mean_rms(store, signals) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """(segment rows with a signal, (rows, 2·channels) mean and RMS of every model channel,
    names). Diagnostics only (the Stage-2 shift and identifiability checks); computed once
    per distinct segment, at the source rate."""
    from rulfuse.core.channels import expand, model_channels

    seg, channels = store.segments, store.channels
    names = mean_rms_names([c.name for c in model_channels(channels)])
    rows = np.flatnonzero(seg["signal_present"].astype(bool))
    out = np.empty((len(rows), len(names)))
    done: dict[int, np.ndarray] = {}
    source, array = None, np.empty((0, len(channels)))
    for i, row in enumerate(rows):
        d = int(seg["distinct_segment"][row])
        if d not in done:
            if seg["source"][row] != source:
                source = seg["source"][row]
                array = np.asarray(signals.read(str(source)))
            start, end = int(seg["start_sample"][row]), int(seg["end_sample"][row])
            x = expand(array[start:end], channels).astype(np.float64)
            done[d] = np.stack([x.mean(axis=0), np.sqrt(np.mean(x**2, axis=0))], 1).reshape(-1)
        out[i] = done[d]
    return rows, out, names
