"""The segment store (PROJECT_PLAN.md §5.8; parts P3-P5 on the stored tables, D9, D14)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import synthetic_dataset_files

from rulfuse.data.dataset import DatasetOptions, load_dataset
from rulfuse.data.segment_store import (
    SegmentStore,
    build_segment_store,
    photo_visible,
    signature,
    signature_payload,
)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    raw, forces = synthetic_dataset_files(root)
    ds = load_dataset(DatasetOptions(), raw=raw, store_dir=forces)
    store = SegmentStore.open(build_segment_store(ds, root / "segments"))
    return root, raw, forces, ds, store


def _tiling_errors(store: SegmentStore) -> list[str]:
    errors = []
    seg, passes = store.segments, store.passes
    for row in np.flatnonzero(passes["recorded"]):
        tool, position = passes["tool"][row], passes["position"][row]
        rows = np.flatnonzero((seg["tool"] == tool) & (seg["position"] == position))
        starts, ends = seg["start_sample"][rows], seg["end_sample"][rows]
        trim_start, trim_end = passes["trim_start"][row], passes["trim_end"][row]
        if starts[0] != trim_start or ends[-1] != trim_end or np.any(starts[1:] != ends[:-1]):
            errors.append(f"{tool} pass {position}: segments do not tile the trimmed pass")
        if np.any((ends[:-1] - trim_start) % 10000 != 0):
            errors.append(f"{tool} pass {position}: an inner boundary is not a multiple of 10 s")
    return errors


def test_segments_tile_every_recorded_pass(synthetic):
    *_, store = synthetic
    assert _tiling_errors(store) == []
    # 23-25 s of cutting: 23 and 24 s give 2 segments (remainder < 5 s merged), 25 s gives 3.
    seg = store.segments
    keys = [f"{t}:{p}" for t, p in zip(seg["tool"], seg["position"], strict=True)]
    per_pass = {key: keys.count(key) for key in set(keys)}
    assert set(per_pass.values()) == {2, 3}


def test_gap_passes_have_rows_but_no_segments(synthetic):
    *_, store = synthetic
    p = store.passes
    gap = (p["tool"] == "T2") & ~p["recorded"]
    assert list(p["position"][gap]) == [9, 10, 11, 12]
    assert list(p["photo_position"][gap]) == [-1] * 4
    seg = store.segments
    assert not np.any((seg["tool"] == "T2") & np.isin(seg["position"], [9, 10, 11, 12]))


def test_duplicates_force_missing_and_shared_encoding(synthetic):
    root, raw, forces, _, store = synthetic
    seg = store.segments
    b3 = seg["record_id"] == "T1R2B3"
    b4 = seg["record_id"] == "T1R2B4"
    assert not seg["force_present"][b3].any() and (seg["distinct_segment"][b3] == -1).all()
    assert seg["force_present"][b4].all() and (seg["representative"][b4] == "T1R2B3").all()
    every = load_dataset(DatasetOptions(attach="every"), raw=raw, store_dir=forces)
    shared = SegmentStore.open(build_segment_store(every, root / "segments-every")).segments
    ids_b3 = shared["distinct_segment"][shared["record_id"] == "T1R2B3"]
    ids_b4 = shared["distinct_segment"][shared["record_id"] == "T1R2B4"]
    assert (ids_b3 >= 0).all() and list(ids_b3) == list(ids_b4)  # one recording, encoded once


def test_targets_match_the_labels_module(synthetic):
    *_, ds, store = synthetic
    seg = store.segments
    for tool, labels in ds.labels.items():
        rows = (seg["tool"] == tool) & seg["is_prediction_point"]
        np.testing.assert_allclose(seg["rul_pass"][rows], labels.targets.rul_pass)
        np.testing.assert_allclose(seg["rul_sec"][rows], labels.targets.rul_sec)
        assert store.tools[tool]["ttf_position"] == labels.ttf_position
        after = (seg["tool"] == tool) & ~seg["is_prediction_point"]
        assert np.all(seg["position"][after] > labels.ttf_position)
        assert np.isnan(seg["rul_pass"][after]).all() and not seg["window_pass"][after].any()


def test_photo_rule_and_baseline_rows(synthetic):
    *_, store = synthetic
    assert not photo_visible(5, 5) and photo_visible(6, 5)
    p = store.passes
    recorded = p["recorded"]
    np.testing.assert_array_equal(p["photo_position"][recorded], p["position"][recorded])
    seg = store.segments
    for tool, info in store.tools.items():
        rows = seg["tool"] == tool
        expected = (
            np.isin(seg["position"][rows], info["baseline_positions"]) & seg["force_present"][rows]
        )
        np.testing.assert_array_equal(seg["in_baseline"][rows], expected)
        assert len(info["baseline_positions"]) == 4


def test_signature_is_stable_and_sensitive(synthetic, tmp_path: Path):
    root, raw, forces, ds, store = synthetic
    again = load_dataset(DatasetOptions(), raw=raw, store_dir=forces)
    assert signature(signature_payload(again)) == store.signature
    copy = SegmentStore.open(build_segment_store(again, tmp_path / "copy"))
    for name, column in store.segments.items():
        if column.dtype.kind == "f":
            np.testing.assert_array_equal(np.nan_to_num(column), np.nan_to_num(copy.segments[name]))
        else:
            np.testing.assert_array_equal(column, copy.segments[name])
    other = load_dataset(DatasetOptions(axis="recorded"), raw=raw, store_dir=forces)
    assert signature(signature_payload(other)) != store.signature
    # An existing store with the same signature is reused, not rewritten.
    stamp = (root / "segments" / "meta.json").stat().st_mtime_ns
    build_segment_store(again, root / "segments")
    assert (root / "segments" / "meta.json").stat().st_mtime_ns == stamp


@pytest.fixture(scope="module")
def real_store(tmp_path_factory, dataset):
    return SegmentStore.open(build_segment_store(dataset, tmp_path_factory.mktemp("real") / "s"))


@pytest.mark.data
def test_real_store_counts_and_tiling(real_store, dataset):
    counts = real_store.meta["counts"]
    assert counts["tools"] == 10 and counts["passes"] == 532 and counts["unrecorded_passes"] == 20
    assert counts["prediction_points"] == 3747
    assert (counts["window_points_pass"], counts["window_points_sec"]) == (1338, 1361)
    assert counts["force_missing_points"] == 951
    assert counts["segments"] == 5125 and counts["distinct_segments"] == 3777
    assert _tiling_errors(real_store) == []
    for tool, labels in dataset.labels.items():
        assert real_store.tools[tool]["ttf_position"] == labels.ttf_position
