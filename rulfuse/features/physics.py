"""Physics features of force segments (PROJECT_PLAN.md §6.2, encoder H).

Stage 2 provides the first part, the per-channel level and spread (mean and RMS of the five
channels), which the P7 validation needs. Stage 3 adds the spectral features of H.
"""

from __future__ import annotations

import numpy as np

from rulfuse.data.channels import CHANNELS, derived_channels, fz_offset
from rulfuse.data.force_store import ForceStore
from rulfuse.data.segment_store import SegmentStore

STAT_NAMES = tuple(f"{stat}_{channel}" for channel in CHANNELS for stat in ("mean", "rms"))


def channel_stats(channels: np.ndarray) -> np.ndarray:
    """(samples, 5) -> 10 values: mean and RMS per channel, in ``STAT_NAMES`` order."""
    x = np.asarray(channels, dtype=np.float64)
    mean = x.mean(axis=0)
    rms = np.sqrt(np.mean(np.square(x), axis=0))
    return np.stack([mean, rms], axis=1).reshape(-1)


def segment_stats(store: SegmentStore, forces: ForceStore) -> tuple[np.ndarray, np.ndarray]:
    """(row indices of force-present segments, (rows, 10) features), computed once per
    distinct segment and shared by duplicated recordings."""
    seg, passes = store.segments, store.passes
    trim_start = {
        (t, int(p)): int(s)
        for t, p, s in zip(passes["tool"], passes["position"], passes["trim_start"], strict=True)
    }
    rows = np.flatnonzero(seg["force_present"])
    by_distinct: dict[int, np.ndarray] = {}
    offsets: dict[tuple[str, int], float] = {}
    out = np.empty((len(rows), len(STAT_NAMES)), dtype=np.float64)
    for index, row in enumerate(rows):
        distinct = int(seg["distinct_segment"][row])
        if distinct not in by_distinct:
            rep = str(seg["representative"][row])
            key = (rep, trim_start[(str(seg["tool"][row]), int(seg["position"][row]))])
            recording = forces.force(rep)
            if key not in offsets:
                offsets[key] = fz_offset(recording[key[1] :])
            start, end = int(seg["start_sample"][row]), int(seg["end_sample"][row])
            by_distinct[distinct] = channel_stats(
                derived_channels(recording[start:end], offsets[key])
            )
        out[index] = by_distinct[distinct]
    return rows, out
