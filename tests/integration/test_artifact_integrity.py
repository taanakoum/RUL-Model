"""P8/P14/P15/P20: corruption, incompatible caches, and misaligned seed members fail closed."""

from __future__ import annotations

import copy
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rulfuse.core.folds import Fold
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.cache import extract_signal_features
from rulfuse.features.encoders import ProjectionEncoder
from rulfuse.gate3 import stack_seeds
from rulfuse.study import Study, ensemble_validation, fit_dir, write_manifest
from rulfuse.tokens import build_fold_data
from tests.factories import MICRO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.adversarial,
    pytest.mark.requirements("P8", "P14", "P15", "P20", "P21"),
]


def prediction_rows():
    return {
        "unit": np.array(["V", "V"]),
        "role": np.array(["validation"] * 2),
        "tau_s": np.array([1.0, 2.0]),
        "segment": np.array([0, 1]),
        "rul_s": np.array([5.0, 4.0]),
        "window": np.array([True, True]),
        "mu_s": np.array([4.0, 3.0]),
        "sigma_s": np.array([1.0, 1.0]),
        "nu": np.array([5.0, 5.0]),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("unit", np.array(["foreign", "foreign"])),
        ("rul_s", np.array([50.0, 40.0])),
        ("window", np.array([False, True])),
    ],
)
@pytest.mark.compatibility
def test_calibration_seed_members_must_agree_on_unit_truth_and_window(field, value):
    first, second = prediction_rows(), prediction_rows()
    second[field] = value
    with pytest.raises(ValueError):
        stack_seeds([first, second])


@pytest.mark.parametrize(
    "field,value",
    [
        ("tau_s", np.array([1.0, 3.0])),
        ("unit", np.array(["foreign", "foreign"])),
        ("rul_s", np.array([50.0, 40.0])),
        ("window", np.array([False, True])),
    ],
)
@pytest.mark.compatibility
def test_main_ensemble_refuses_misaligned_seed_tables(tmp_path, field, value):
    fold = Fold("T", "V", ("A", "B"))
    for seed in (0, 1):
        rows = prediction_rows()
        if seed:
            rows[field] = value
        folder = fit_dir(tmp_path, "a", fold, seed)
        folder.mkdir(parents=True)
        pq.write_table(pa.table(rows), folder / "predictions.parquet")
        (folder / "DONE").write_text("ok", encoding="utf-8")
    with pytest.raises(ValueError):
        ensemble_validation(tmp_path, "a", fold, [0, 1])


@pytest.mark.parametrize("corruption", ["foreign-store", "incomplete-signal", "incomplete-image"])
def test_token_building_refuses_foreign_or_unfinished_features(micro_bundle, corruption):
    b = micro_bundle
    signal, image = copy.copy(b.signal), copy.copy(b.image)
    assert image is not None
    if corruption == "foreign-store":
        signal.meta = copy.deepcopy(signal.meta)
        signal.meta["signature"]["store"] = "another-dataset"
    elif corruption == "incomplete-signal":
        signal.done = np.zeros_like(signal.done)
    else:
        image.done = np.zeros_like(image.done)
    with pytest.raises(ValueError):
        build_fold_data(b.store, signal, image, b.folds[0], MICRO)


def test_nonfinite_encoder_outputs_cannot_be_published_as_a_complete_cache(micro_bundle, tmp_path):
    class BrokenEncoder(ProjectionEncoder):
        def encode(self, x):
            return np.full((len(x), self.dim), np.nan, np.float32)

    b = micro_bundle
    with pytest.raises(ValueError):
        extract_signal_features(
            b.store,
            b.dataset.signals,
            BrokenEncoder(dim=4),
            root=tmp_path / "broken",
            log=lambda *_: None,
        )


def test_failed_store_rebuild_preserves_previous_complete_store(
    micro_bundle, tmp_path, monkeypatch
):
    b = micro_bundle
    root = build_segment_store(b.dataset, tmp_path / "store")
    original = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    write = pq.write_table
    count = 0

    def fail_second_write(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated disk failure")
        return write(*args, **kwargs)

    monkeypatch.setattr(pq, "write_table", fail_second_write)
    with pytest.raises(OSError, match="disk failure"):
        build_segment_store(b.dataset, root, force=True)
    assert original == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    assert SegmentStore.open(root).signature == b.store.signature


def test_an_incomplete_gate_cannot_supply_the_next_studys_configuration(tmp_path):
    (tmp_path / "earlier").mkdir()
    (tmp_path / "earlier/decision.json").write_text(
        json.dumps({"complete": False, "adopted": {"lr": 0.003}}),
        encoding="utf-8",
    )
    study = Study(
        "next", {"inherits": ["earlier"], "arms": {"a": {}}, "decision": {"type": "none"}}
    )
    with pytest.raises((ValueError, RuntimeError)):
        write_manifest(study, ["A", "B", "C"], tmp_path)
    assert not (tmp_path / "next/manifest.tsv").exists()


def test_unknown_fold_names_cannot_silently_create_an_empty_study(tmp_path):
    study = Study("bad-fold", {"folds": ["typo"], "arms": {"a": {}}, "decision": {"type": "none"}})
    with pytest.raises(ValueError):
        write_manifest(study, ["A", "B", "C"], tmp_path)
