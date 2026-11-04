"""Force channels (D4, §5.4) and the first physics features (§6.2 H)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import synthetic_dataset_files

from rulfuse.data.channels import CHANNELS, derived_channels, fz_offset
from rulfuse.data.dataset import DatasetOptions, load_dataset
from rulfuse.data.segment_store import SegmentStore, build_segment_store
from rulfuse.features.physics import STAT_NAMES, channel_stats, segment_stats


def test_derived_channels():
    force = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, -2.0]], np.float32)
    out = derived_channels(force, offset=-1.0)
    assert out.shape == (2, len(CHANNELS))
    np.testing.assert_allclose(out[:, 3], [1.0, -1.0])  # Fz - offset
    np.testing.assert_allclose(out[:, 4], [5.0, 2.0])  # |F|


def test_fz_offset_uses_only_the_first_second():
    rng = np.random.default_rng(0)
    trimmed = rng.normal(size=(5000, 3)).astype(np.float32)
    trimmed[:1000, 2] = -7.0
    assert fz_offset(trimmed) == pytest.approx(-7.0)
    changed = trimmed.copy()
    changed[1000:, 2] = 1e6  # later samples never change the offset (causal)
    assert fz_offset(changed) == fz_offset(trimmed)


def test_channel_stats_by_hand():
    x = np.array([[1.0, -2.0, 0.0, 0.0, 3.0], [3.0, 2.0, 0.0, 0.0, 4.0]])
    stats = dict(zip(STAT_NAMES, channel_stats(x), strict=True))
    assert stats["mean_Fx"] == 2.0 and stats["rms_Fx"] == pytest.approx(np.sqrt(5.0))
    assert stats["mean_Fy"] == 0.0 and stats["rms_Fy"] == pytest.approx(2.0)
    assert stats["rms_F"] == pytest.approx(np.sqrt(12.5))


def test_segment_stats_cover_force_present_segments(tmp_path):
    raw, forces = synthetic_dataset_files(tmp_path)
    ds = load_dataset(DatasetOptions(attach="every"), raw=raw, store_dir=forces)
    store = SegmentStore.open(build_segment_store(ds, tmp_path / "s"))
    rows, features = segment_stats(store, ds.store)
    assert np.all(store.segments["force_present"][rows])
    assert features.shape == (len(rows), len(STAT_NAMES)) and np.isfinite(features).all()
    seg = store.segments
    b3 = np.flatnonzero(seg["record_id"][rows] == "T1R2B3")
    b4 = np.flatnonzero(seg["record_id"][rows] == "T1R2B4")
    np.testing.assert_array_equal(features[b3], features[b4])  # one recording, one result
    mean_fx = features[:, STAT_NAMES.index("mean_Fx")]
    assert np.all((mean_fx > 90) & (mean_fx < 160))  # cutting level, idle ends trimmed away
