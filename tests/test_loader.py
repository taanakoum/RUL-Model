"""P1 Loading (PROJECT_PLAN.md §15): ids, labels, the MAT table, duplicate groups, the store."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from helpers import label_row, write_label_csvs

from rulfuse import paths
from rulfuse.adapters.nonastreda.force_store import ForceStore, sha256_file
from rulfuse.adapters.nonastreda.load import (
    decode_record_id,
    duplicate_groups,
    load_labels,
    parse_datastore,
    read_csv,
)
from rulfuse.adapters.nonastreda.verify import verify_data

OMNIFUSER_RECORDS = (
    paths.REPO_ROOT.parent / "OmniFuser" / "experiment_results" / "eda" / "records.csv"
)


def test_decode_record_id_accepts_file_names():
    assert decode_record_id("T10R3B4.jpg") == ("T10", 3, 4, "T10R3B4")
    assert decode_record_id("T1R12B1") == ("T1", 12, 1, "T1R12B1")
    with pytest.raises(ValueError):
        decode_record_id("X1R1B1")


def test_read_csv_rejects_duplicates_and_missing_id(tmp_path: Path):
    (tmp_path / "a.csv").write_text("id,x\nT1R1B1,1\nT1R1B1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_csv(tmp_path / "a.csv")
    (tmp_path / "b.csv").write_text("name,x\nT1R1B1,1\n")
    with pytest.raises(ValueError, match="id column"):
        read_csv(tmp_path / "b.csv")


def test_load_labels_orders_and_validates(tmp_path: Path):
    write_label_csvs(tmp_path, [label_row("T2R1B1"), label_row("T1R2B1"), label_row("T1R1B2")])
    metas = load_labels(tmp_path)
    assert [m.record_id for m in metas] == ["T1R1B2", "T1R2B1", "T2R1B1"]
    write_label_csvs(tmp_path, [label_row("T1R1B1", label="blunt")])
    with pytest.raises(ValueError, match="unknown image label"):
        load_labels(tmp_path)
    write_label_csvs(tmp_path, [label_row("T1R1B1", tool_label="4")])
    with pytest.raises(ValueError, match="tool label"):
        load_labels(tmp_path)
    write_label_csvs(tmp_path, [label_row("T1R1B1", wear="nan")])
    with pytest.raises(ValueError, match="non-finite"):
        load_labels(tmp_path)


def test_load_labels_requires_matching_id_sets(tmp_path: Path):
    write_label_csvs(tmp_path, [label_row("T1R1B1")])
    (tmp_path / "labels_reg.csv").write_text("id,gaps,flank_wear,overhang\nT1R1B2,0,1,0\n")
    with pytest.raises(ValueError, match="must match"):
        load_labels(tmp_path)


def _datastore(rows: list[tuple[str, str, np.ndarray]]) -> np.ndarray:
    table = np.empty((len(rows), 4), dtype=object)
    for index, (name, label, force) in enumerate(rows):
        table[index] = [name, label, None, force]
    return table


def test_parse_datastore_converts_and_checks(tmp_path: Path):
    write_label_csvs(tmp_path, [label_row("T1R1B1"), label_row("T1R1B2", label="used")])
    metas = load_labels(tmp_path)
    force = np.arange(12, dtype=np.float64).reshape(3, 4)
    parsed = dict(
        parse_datastore(_datastore([("T1R1B1", "sharp", force), ("T1R1B2", "used", force)]), metas)
    )
    assert parsed["T1R1B1"].shape == (4, 3) and parsed["T1R1B1"].dtype == np.float32
    np.testing.assert_array_equal(parsed["T1R1B1"], force.T)
    with pytest.raises(ValueError, match="label mismatch"):
        list(
            parse_datastore(
                _datastore([("T1R1B1", "dulled", force), ("T1R1B2", "used", force)]), metas
            )
        )
    with pytest.raises(ValueError, match="shape"):
        list(parse_datastore(_datastore([("T1R1B1", "sharp", force[:2])]), metas))
    bad = force.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        list(parse_datastore(_datastore([("T1R1B1", "sharp", bad)]), metas))
    with pytest.raises(ValueError, match="without force"):
        list(parse_datastore(_datastore([("T1R1B1", "sharp", force)]), metas))


def test_duplicate_groups_within_runs_only():
    a, b = np.ones((5, 3), np.float32), np.zeros((5, 3), np.float32)
    forces = {"T1R1B1": a, "T1R1B3": b, "T1R1B4": b.copy(), "T1R2B1": a.copy(), "T2R1B1": b.copy()}
    assert duplicate_groups(forces) == [("T1R1B3", "T1R1B4")]


@pytest.mark.data
def test_real_labels_and_store(metas):
    assert len(metas) == 512
    per_tool = Counter(m.tool for m in metas)
    assert per_tool == {
        "T1": 40,
        "T2": 56,
        "T3": 52,
        "T4": 40,
        "T5": 48,
        "T6": 60,
        "T7": 56,
        "T8": 52,
        "T9": 52,
        "T10": 56,
    }
    store = ForceStore.open(paths.force_store_dir())
    assert store.meta["records"] == 512
    assert store.meta["distinct_recordings"] == 378
    assert len(store.groups) == 132
    patterns = Counter(tuple(decode_record_id(r)[2] for r in group) for group in store.groups)
    assert patterns == {(3, 4): 127, (1, 2): 4, (1, 2, 3, 4): 1}
    assert sum(len(group) - 1 for group in store.groups) == 134
    raw_mat = paths.raw_dir() / "forces_xyz_raw.mat"
    assert store.meta["source_sha256"] == sha256_file(raw_mat)


@pytest.mark.data
def test_store_matches_omnifuser_dedup():
    """OmniFuser keeps one member per group (largest wear); its retained set must equal ours."""
    if not OMNIFUSER_RECORDS.is_file():
        pytest.skip("OmniFuser's records.csv is not available")
    rows = list(csv.DictReader(OMNIFUSER_RECORDS.open()))
    retained = {r["record_id"] for r in rows if r["retained_after_dedup"] == "True"}
    store = ForceStore.open(paths.force_store_dir())
    assert len(retained) == 378
    # Every OmniFuser group (same tool, run and force sum) is one of ours.
    ours = {frozenset(group) for group in store.groups}
    by_key: dict[tuple, set[str]] = {}
    for r in rows:
        by_key.setdefault((r["tool"], r["run"], r["samples"], r["magnitude_sum"]), set()).add(
            r["record_id"]
        )
    theirs = {frozenset(ids) for ids in by_key.values() if len(ids) > 1}
    assert ours == theirs


@pytest.mark.data
def test_verify_data_report():
    report = verify_data(paths.raw_dir(), paths.REPO_ROOT / "Data")
    assert report["ok"], json.dumps(report)[:500]
    assert report["files_expected"] == 1539
    assert report["run_gaps"]["T4"]["interior"] == [6, 7, 8, 9]
    assert report["run_gaps"]["T2"]["interior"] == [10]
    assert report["run_gaps"]["T1"]["leading"] == [1]


def test_verify_data_on_synthetic_copy(tmp_path: Path):
    raw = tmp_path / "raw"
    write_label_csvs(raw, [label_row(f"T1R1B{b}") for b in range(1, 5)])
    for name in ("labels.csv", "labels_reg.csv"):  # the publisher's files end lines with LF
        path = raw / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    for kind, ext in (("tool", ".jpg"), ("chip", ".jpg"), ("work", ".png")):
        (raw / kind).mkdir()
        for b in range(1, 5):
            (raw / kind / f"T1R1B{b}{ext}").write_bytes(f"{kind}{b}".encode())
    files = {}
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        rel = path.relative_to(raw).as_posix()
        files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    (raw / "MANIFEST.json").write_text(json.dumps({"files": files}))
    repo = tmp_path / "repo"
    write_label_csvs(repo, [label_row(f"T1R1B{b}") for b in range(1, 5)])
    crlf = (repo / "labels.csv").read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    (repo / "labels.csv").write_bytes(crlf)
    report = verify_data(raw, repo, expected_passes=4)
    assert report["ok"], report
    assert report["repo_data"]["line_endings_only"] == ["labels.csv", "labels_reg.csv"]
    (raw / "tool" / "T1R1B2.jpg").write_bytes(b"changed")
    report = verify_data(raw, repo, expected_passes=4)
    assert not report["ok"] and report["files_mismatched"] == ["tool/T1R1B2.jpg"]
