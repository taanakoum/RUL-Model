"""Read-only acceptance checks against docs/EDA_RUL.md, sections 3 and 8.

These pin the approved, gap-aware 10 s dataset; they never read experiment predictions.
"""

from __future__ import annotations

import numpy as np
import pytest

from rulfuse import paths
from rulfuse.core.store import SegmentStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.data,
    pytest.mark.requirements("P1", "P2", "P3", "P4", "P5", "P8"),
]


@pytest.fixture(scope="module")
def approved_store():
    root = paths.interim_dir() / "segments/nonastreda/3e357b5a23ff"
    if not (root / "meta.json").is_file():
        pytest.skip(
            "approved default segment store unavailable; build it before acceptance testing"
        )
    return SegmentStore.open(root)


def test_approved_eda_counts_and_thin_window_have_not_drifted(approved_store):
    store = approved_store
    expected = {
        "T1": (308, 90, 155),
        "T2": (393, 99, 116),
        "T3": (355, 89, 156),
        "T4": (216, 49, 20),
        "T5": (188, 50, 156),
        "T6": (458, 118, 154),
        "T7": (465, 119, 155),
        "T8": (459, 110, 154),
        "T9": (379, 96, 148),
        "T10": (526, 131, 147),
    }
    assert set(store.unit_names) == set(expected)
    assert len(store.recordings["unit"]) == 512
    assert len(store.images["unit"]) == 1536
    for unit, (points, missing, window) in expected.items():
        rows = store.segment_rows(unit)
        segments = store.segments
        prediction = segments["is_prediction_point"][rows].astype(bool)
        assert prediction.sum() == points, unit
        assert (prediction & ~segments["signal_present"][rows].astype(bool)).sum() == missing, unit
        assert segments["window"][rows].sum() == window, unit
    assert sum(v[0] for v in expected.values()) == 3747
    assert sum(v[1] for v in expected.values()) == 951


def test_approved_failure_times_and_photo_availability_match_eda(approved_store):
    store = approved_store
    failures = dict(
        zip(
            [f"T{i}" for i in range(1, 11)],
            [3000, 4203, 3463, 3657, 1827, 4472, 4543, 4474, 3837, 5367],
            strict=True,
        )
    )
    for unit, rounded_time in failures.items():
        assert abs(store.units[unit]["failure_s"] - rounded_time) < 1.0
    end = dict(zip(store.recordings["key"], store.recordings["end_s"], strict=True))
    np.testing.assert_allclose(
        store.images["available_s"], [end[key] for key in store.images["key"]], rtol=0, atol=1e-8
    )
    kinds, counts = np.unique(store.images["kind"], return_counts=True)
    assert len(kinds) == 3 and (counts == 512).all()
