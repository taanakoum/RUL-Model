"""P8 Feature caches (PROJECT_PLAN.md §5.8): rows, duplicates, resume, signatures."""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import synthetic_dataset_files

from rulfuse.adapters.nonastreda import NonastredaOptions, load
from rulfuse.core.channels import descriptors, expand
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import (
    distinct_rows,
    extract_image_features,
    extract_signal_features,
    find_features,
    open_signal_features,
    resample,
    windows,
)
from rulfuse.features.encoders import PixelStatsEncoder, ProjectionEncoder


def quiet(*_: object) -> None:
    return None


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("cache")
    raw, forces = synthetic_dataset_files(root)
    ds = load(NonastredaOptions(attach="every"), raw=raw, store_dir=forces)
    store = SegmentStore.open(build_segment_store(ds, root / "segments"))
    return root, ds, store


def test_windows_and_resampling():
    x = np.arange(1100, dtype=np.float32)[:, None].repeat(2, axis=1)
    w = windows(x, 512)
    assert w.shape == (2, 3, 512)
    assert w[0, 2, -1] == 1099 and w[0, 1, 0] == 512  # last window aligned to the end
    assert windows(x[:100], 512).shape == (2, 1, 512)  # short: edge-padded
    y = resample(np.sin(np.arange(4000) / 20.0)[:, None], 2000.0, 1000.0)
    assert y.shape == (2000, 1)


def test_signal_cache_rows_duplicates_and_content(synthetic):
    root, ds, store = synthetic
    features = extract_signal_features(
        store, ds.signals, ProjectionEncoder(), root=root / "f", log=quiet
    )
    n = int(store.segments["distinct_segment"].max()) + 1
    assert features.complete and features.embeddings.shape == (n, 4, 32)
    seg = store.segments
    b3 = seg["distinct_segment"][seg["recording_key"] == "T1R2B3"]
    b4 = seg["distinct_segment"][seg["recording_key"] == "T1R2B4"]
    assert list(b3) == list(b4)  # one recording, one row: duplicates are encoded once
    # Descriptors of one row recomputed by hand from the force store.
    d = 5
    row = distinct_rows(store)[d]
    array = ds.signals.read(str(seg["source"][row]))
    start, end = int(seg["start_sample"][row]), int(seg["end_sample"][row])
    head_rows = array[seg["recording_start_sample"][row] :][:1000]
    head = expand(head_rows, ds.channels).mean(axis=0, dtype=np.float64)
    expected = descriptors(expand(array[start:end], ds.channels), 1000.0, head)
    np.testing.assert_allclose(features.descriptors[d], expected, rtol=1e-5, atol=1e-5)


def test_resume_gives_the_same_cache_and_foreign_signatures_are_refused(synthetic, tmp_path):
    _, ds, store = synthetic
    encoder = ProjectionEncoder()
    once = extract_signal_features(store, ds.signals, encoder, root=tmp_path / "a", log=quiet)
    part = extract_signal_features(
        store, ds.signals, encoder, root=tmp_path / "b", limit=7, log=quiet
    )
    assert not part.complete and int(part.done.sum()) == 7
    resumed = extract_signal_features(store, ds.signals, encoder, root=tmp_path / "b", log=quiet)
    np.testing.assert_array_equal(np.asarray(once.embeddings), np.asarray(resumed.embeddings))
    np.testing.assert_array_equal(np.asarray(once.descriptors), np.asarray(resumed.descriptors))
    other = ProjectionEncoder(seed=3)
    assert (
        extract_signal_features(store, ds.signals, other, root=tmp_path / "b", log=quiet).folder
        != resumed.folder
    )
    meta = json.loads((resumed.folder / "meta.json").read_text())
    meta["signature"]["model_rate_hz"] = 123.0  # tamper: the folder now holds "other" features
    (resumed.folder / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="other features"):
        extract_signal_features(store, ds.signals, encoder, root=tmp_path / "b", log=quiet)
    with pytest.raises(ValueError, match="signature mismatch"):
        open_signal_features(store, encoder.signature(), root=tmp_path / "b")


def test_image_cache_and_lookup(synthetic):
    root, _, store = synthetic
    features = extract_image_features(
        store, PixelStatsEncoder(), root=root / "f", batch=5, log=quiet
    )
    assert features.complete and features.tokens.shape == (len(store.images["unit"]), 9, 6)
    brightness = np.asarray(features.tokens[store.image_rows("T1", "tool"), 0, 0], dtype=float)
    assert np.all(np.diff(brightness) >= -1e-3)  # the synthetic photos brighten with wear
    assert find_features(store, "image", "pixelstats", root=root / "f") == features.folder
