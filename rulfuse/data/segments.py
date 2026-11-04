"""Segment boundaries inside a pass (PROJECT_PLAN.md D7, §5.5).

A pass is cut into 10 s windows from its trimmed start. A remainder shorter than 5 s joins
the previous segment; a pass shorter than one segment is one segment. Each segment end is a
prediction point.
"""

from __future__ import annotations

import numpy as np

SEGMENT_SECONDS = 10.0
MIN_REMAINDER_SECONDS = 5.0


def segment_ends(
    duration_s: float,
    segment_s: float = SEGMENT_SECONDS,
    min_remainder_s: float = MIN_REMAINDER_SECONDS,
) -> np.ndarray:
    """Elapsed seconds (from the trimmed start) at the end of each segment of a pass."""
    if duration_s <= 0:
        raise ValueError("a pass must have a positive duration")
    if segment_s <= 0 or not 0 <= min_remainder_s <= segment_s:
        raise ValueError("need segment_s > 0 and 0 <= min_remainder_s <= segment_s")
    full = int(np.floor(duration_s / segment_s + 1e-9))
    if full == 0:
        return np.array([duration_s])
    ends = [segment_s * (index + 1) for index in range(full)]
    remainder = duration_s - full * segment_s
    if remainder >= min_remainder_s:
        ends.append(duration_s)
    else:
        ends[-1] = duration_s
    return np.asarray(ends, dtype=np.float64)


def segment_sample_bounds(
    samples: int,
    rate_hz: float = 1000.0,
    segment_s: float = SEGMENT_SECONDS,
    min_remainder_s: float = MIN_REMAINDER_SECONDS,
) -> list[tuple[int, int]]:
    """(start, end) sample indices of each segment of a trimmed pass of ``samples`` samples."""
    ends = segment_ends(samples / rate_hz, segment_s, min_remainder_s)
    stops = [int(round(end * rate_hz)) for end in ends]
    stops[-1] = samples
    starts = [0, *stops[:-1]]
    return list(zip(starts, stops, strict=True))
