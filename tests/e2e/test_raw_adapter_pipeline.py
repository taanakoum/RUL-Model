"""P1-P5/P8/P14/P22: MATLAB and CSV -> raw store -> preprocessing -> cached features -> fit."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy.io import savemat

from rulfuse.adapters.nonastreda import load
from rulfuse.adapters.nonastreda.force_store import ForceStore, build_force_store
from rulfuse.adapters.nonastreda.load import MAT_FILE, load_labels
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import extract_signal_features
from rulfuse.features.encoders import ProjectionEncoder
from rulfuse.train import fit
from tests.factories import MICRO
from tests.helpers import synthetic_dataset_files

pytestmark = [pytest.mark.e2e,
              pytest.mark.requirements("P1", "P2", "P3", "P4", "P5", "P8", "P14", "P22")]


def test_mat_csv_adapter_roundtrip_preserves_gaps_duplicates_truth_and_fit(tmp_path):
    raw, original_root = synthetic_dataset_files(tmp_path / "raw-data")
    original = ForceStore.open(original_root)
    metas = load_labels(raw)
    matlab = np.empty((len(metas), 4), dtype=object)
    for i, meta in enumerate(metas):
        matlab[i] = [meta.record_id, meta.image_label, "unused", original.force(meta.record_id).T]
    savemat(raw / MAT_FILE, {"baseDatastore": matlab})
    rebuilt = ForceStore.open(build_force_store(raw, tmp_path / "rebuilt-forces", metas))
    assert rebuilt.groups == original.groups
    for meta in metas:
        np.testing.assert_array_equal(rebuilt.force(meta.record_id), original.force(meta.record_id))
    dataset = load(raw=raw, store_dir=rebuilt.root)
    store = SegmentStore.open(build_segment_store(dataset, tmp_path / "segments"))
    assert len(store.units) == 3
    assert store.meta["counts"]["recordings"] == 36
    assert store.meta["counts"]["signal_missing_points"] > 0
    for unit in store.unit_names:
        rows = store.segment_rows(unit)
        points = store.segments["is_prediction_point"][rows].astype(bool)
        tau = store.segments["tau_s"][rows].astype(float)
        np.testing.assert_allclose(store.segments["rul_s"][rows][points],
                                   store.units[unit]["failure_s"] - tau[points])
    signal = extract_signal_features(store, dataset.signals, ProjectionEncoder(dim=4),
                                     root=tmp_path / "features", log=lambda *_: None)
    fold = leave_one_unit_out(store.unit_names)[0]
    config = MICRO.with_overrides({"players": ("signal",)})
    result = fit(store, signal, None, fold, 0, config, tmp_path / "fit", log=lambda *_: None)
    assert result["selected_step"] is not None and (tmp_path / "fit/DONE").is_file()
    predictions = pq.read_table(tmp_path / "fit/predictions.parquet").to_pydict()
    assert np.isfinite(predictions["mu_s"]).all()
