"""P4 Segmentation (PROJECT_PLAN.md D7, §15)."""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse.core.segments import segment_ends, segment_sample_bounds, subwindow_starts


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (97.1, [10, 20, 30, 40, 50, 60, 70, 80, 90, 97.1]),
        (94.9, [10, 20, 30, 40, 50, 60, 70, 80, 94.9]),
        (100.0, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        (15.0, [10, 15]),
        (14.9, [14.9]),
        (8.0, [8.0]),
    ],
)
def test_segment_ends_remainder_rule(duration, expected):
    np.testing.assert_allclose(segment_ends(duration), expected)


def test_segment_ends_rejects_bad_input():
    with pytest.raises(ValueError):
        segment_ends(0.0)
    with pytest.raises(ValueError):
        segment_ends(50.0, segment_s=10, min_remainder_s=11)


def test_sample_bounds_tile_the_pass():
    for samples in (8000, 14900, 97123, 100000):
        bounds = segment_sample_bounds(samples, 1000.0)
        assert bounds[0][0] == 0 and bounds[-1][1] == samples
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
        # Every boundary but the last is a whole multiple of 10 s: it depends only on the
        # time since the trimmed start, never on the pass length.
        assert all(stop % 10000 == 0 for _, stop in bounds[:-1])


def test_subwindows_cover_every_sample():
    for samples in (512, 513, 10000, 14321):
        starts = subwindow_starts(samples, 512)
        covered = np.zeros(samples, dtype=bool)
        for a in starts:
            covered[a : a + 512] = True
        assert covered.all() and starts[-1] + 512 == samples and starts[0] == 0
    assert len(subwindow_starts(10000, 512)) == 20  # a 10 s segment at 1 kHz (§6.2)
    assert subwindow_starts(300, 512).tolist() == [0]  # shorter: one window, padded later


@pytest.mark.data
def test_real_point_counts(real_store):
    seg = real_store.segments
    point = seg["is_prediction_point"].astype(bool)
    counts = {u: int((point & (seg["unit"] == u)).sum()) for u in real_store.unit_names}
    assert counts == {
        "T1": 308,
        "T2": 393,
        "T3": 355,
        "T4": 216,
        "T5": 188,
        "T6": 458,
        "T7": 465,
        "T8": 459,
        "T9": 379,
        "T10": 526,
    }
