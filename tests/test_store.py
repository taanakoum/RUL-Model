"""The generic segment store built through the Nonastreda adapter (PROJECT_PLAN.md §5.8;
parts P3-P5 on the stored tables, D9, D14)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import synthetic_dataset_files

from rulfuse.adapters.nonastreda import NonastredaOptions, load
from rulfuse.core.store import (
    SegmentStore,
    build_segment_store,
    image_visible,
    signature_payload,
    stable_hash,
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    raw, forces = synthetic_dataset_files(root)
    ds = load(NonastredaOptions(), raw=raw, store_dir=forces)
    store = SegmentStore.open(build_segment_store(ds, root / "segments"))
    return root, raw, forces, ds, store


def _tiling_errors(store: SegmentStore, rate: float) -> list[str]:
    errors = []
    seg, rec = store.segments, store.recordings
    for row in np.flatnonzero(rec["signal_present"]):
        unit, index = rec["unit"][row], rec["index"][row]
        rows = np.flatnonzero((seg["unit"] == unit) & (seg["recording"] == index))
        starts, ends = seg["start_sample"][rows], seg["end_sample"][rows]
        first, last = rec["start_sample"][row], rec["end_sample"][row]
        if starts[0] != first or ends[-1] != last or np.any(starts[1:] != ends[:-1]):
            errors.append(f"{unit} recording {index}: segments do not tile it")
        if np.any((ends[:-1] - first) % int(10 * rate) != 0):
            errors.append(f"{unit} recording {index}: an inner boundary is not a multiple of 10 s")
    return errors


def test_segments_tile_every_recording(synthetic):
    *_, store = synthetic
    assert _tiling_errors(store, 1000.0) == []
    # 23-25 s of cutting: 23 and 24 s give 2 segments (remainder < 5 s merged), 25 s gives 3.
    seg = store.segments
    keys = [f"{u}:{r}" for u, r in zip(seg["unit"], seg["recording"], strict=True)]
    assert set({key: keys.count(key) for key in set(keys)}.values()) == {2, 3}


def test_gaps_only_shift_operating_time(synthetic):
    *_, ds, store = synthetic
    rec = store.recordings
    t2 = np.flatnonzero(rec["unit"] == "T2")
    assert len(t2) == 12  # 3 recorded runs; the unrecorded run 3 has no recording
    starts, ends = rec["start_s"][t2], rec["end_s"][t2]
    jumps = starts[1:] - ends[:-1]
    assert np.all(jumps[:7] < 1e-9) and jumps[7] > 50  # 4 unrecorded passes between B4 and R4B1
    assert all(r.attrs["run"] in (1, 2, 4) for r in ds.unit("T2").recordings)


def test_duplicates_signal_missing_and_shared_encoding(synthetic):
    root, raw, forces, _, store = synthetic
    seg = store.segments
    b3, b4 = seg["recording_key"] == "T1R2B3", seg["recording_key"] == "T1R2B4"
    assert not seg["signal_present"][b3].any() and (seg["distinct_segment"][b3] == -1).all()
    assert seg["signal_present"][b4].all() and (seg["source"][b4] == "T1R2B3").all()
    every = load(NonastredaOptions(attach="every"), raw=raw, store_dir=forces)
    shared = SegmentStore.open(build_segment_store(every, root / "segments-every")).segments
    ids_b3 = shared["distinct_segment"][shared["recording_key"] == "T1R2B3"]
    ids_b4 = shared["distinct_segment"][shared["recording_key"] == "T1R2B4"]
    assert (ids_b3 >= 0).all() and list(ids_b3) == list(ids_b4)  # one recording, encoded once


def test_targets_match_the_failure_times(synthetic):
    *_, ds, store = synthetic
    seg = store.segments
    for unit in ds.units:
        rows = (seg["unit"] == unit.name) & seg["is_prediction_point"]
        assert unit.failure_s is not None
        np.testing.assert_allclose(seg["rul_s"][rows], unit.failure_s - seg["tau_s"][rows])
        after = (seg["unit"] == unit.name) & ~seg["is_prediction_point"]
        assert np.all(seg["tau_s"][after] > unit.failure_s)
        assert np.isnan(seg["rul_s"][after]).all() and not seg["window"][after].any()
        assert store.units[unit.name]["failure_recording"] == unit.attrs["ttf_record"]


def test_image_rule_and_baseline_rows(synthetic):
    *_, store = synthetic
    assert not image_visible(100.0, 100.0) and image_visible(100.0, 100.01)
    rec, img, seg = store.recordings, store.images, store.segments
    end_of = {(u, k): e for u, k, e in zip(rec["unit"], rec["key"], rec["end_s"], strict=True)}
    for u, k, a in zip(img["unit"], img["key"], img["available_s"], strict=True):
        assert a == end_of[(u, k)]  # a photo is available when its pass ends (D9)
    # A point never sees its own recording's photos, and sees every earlier one.
    for row in range(len(seg["unit"])):
        own = (img["unit"] == seg["unit"][row]) & (img["key"] == seg["recording_key"][row])
        assert not image_visible(img["available_s"][own], seg["tau_s"][row]).any()
    for unit, info in store.units.items():
        rows = seg["unit"] == unit
        expected = (
            np.isin(seg["recording_key"][rows], info["baseline_recordings"])
            & seg["signal_present"][rows]
        )
        np.testing.assert_array_equal(seg["in_baseline"][rows], expected)
        assert len(info["baseline_recordings"]) == 4


def test_signature_is_stable_and_sensitive(synthetic, tmp_path: Path):
    root, raw, forces, ds, store = synthetic
    again = load(NonastredaOptions(), raw=raw, store_dir=forces)
    assert stable_hash(signature_payload(again)) == store.signature
    copy = SegmentStore.open(build_segment_store(again, tmp_path / "copy"))
    for name, column in store.segments.items():
        if column.dtype.kind == "f":
            np.testing.assert_array_equal(np.nan_to_num(column), np.nan_to_num(copy.segments[name]))
        else:
            np.testing.assert_array_equal(column, copy.segments[name])
    other = load(NonastredaOptions(axis="recorded"), raw=raw, store_dir=forces)
    assert stable_hash(signature_payload(other)) != store.signature
    stamp = (root / "segments" / "meta.json").stat().st_mtime_ns
    build_segment_store(again, root / "segments")  # same signature: reused, not rewritten
    assert (root / "segments" / "meta.json").stat().st_mtime_ns == stamp


@pytest.mark.data
def test_real_store_counts_and_tiling(real_store, data):
    counts = real_store.meta["counts"]
    assert counts["units"] == 10 and counts["recordings"] == 512
    assert counts["prediction_points"] == 3747 and counts["window_points"] == 1361
    assert counts["signal_missing_points"] == 951
    assert counts["segments"] == 5125 and counts["distinct_segments"] == 3777
    assert counts["images_by_kind"] == {"chip": 512, "tool": 512, "work": 512}
    assert _tiling_errors(real_store, 1000.0) == []
    for tool, failure in data.failures.items():
        assert real_store.units[tool]["failure_recording"] == failure.ttf_record
        assert real_store.units[tool]["failure_s"] == pytest.approx(failure.tau_ttf)
    t4 = real_store.segments["unit"] == "T4"
    assert int(real_store.segments["window"][t4].sum()) == 20  # T4: 20 window points (§2.2)
