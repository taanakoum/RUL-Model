"""Raw force store: every distinct recording once, float32, memory-mapped (PROJECT_PLAN.md §5.8).

The MAT file takes about 1.2 GB once decoded and loads slowly, so it is read once by
``build_force_store``. Later steps open ``signals.npy`` memory-mapped. Duplicated recordings
(§2.4) are stored once; every record id of a group points at the same rows.

Layout of ``<root>``:

* ``signals.npy``: (total samples, 3) float32, untrimmed Fx, Fy, Fz of the distinct recordings;
* ``index.csv``: record_id, representative (first member of its group), offset, samples;
* ``groups.json``: the duplicate groups, members in block order;
* ``meta.json``: store version, source file size and sha256, counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rulfuse.data.nonastreda import (
    MAT_FILE,
    PassMeta,
    duplicate_groups,
    parse_datastore,
    pass_order,
    read_datastore,
)

STORE_VERSION = 1


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ForceStore:
    root: Path
    signals: np.ndarray
    offsets: dict[str, tuple[int, int]]
    representative: dict[str, str]
    groups: tuple[tuple[str, ...], ...]
    meta: dict

    @classmethod
    def open(cls, root: str | Path) -> ForceStore:
        path = Path(root)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("store_version") != STORE_VERSION:
            raise ValueError(
                f"{path}: store version {meta.get('store_version')} != {STORE_VERSION}"
            )
        signals = np.load(path / "signals.npy", mmap_mode="r")
        offsets: dict[str, tuple[int, int]] = {}
        representative: dict[str, str] = {}
        with (path / "index.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                offsets[row["record_id"]] = (int(row["offset"]), int(row["samples"]))
                representative[row["record_id"]] = row["representative"]
        groups = tuple(tuple(group) for group in json.loads((path / "groups.json").read_text()))
        return cls(path, signals, offsets, representative, groups, meta)

    def force(self, record_id: str) -> np.ndarray:
        """Untrimmed (samples, 3) float32 force of a record (a read-only view)."""
        offset, samples = self.offsets[record_id]
        return self.signals[offset : offset + samples]

    def record_ids(self) -> list[str]:
        return sorted(self.offsets, key=pass_order)

    def distinct_ids(self) -> list[str]:
        """One record id per distinct recording (the first member of each group)."""
        return sorted({rep for rep in self.representative.values()}, key=pass_order)


def build_force_store(raw: str | Path, root: str | Path, metas: Sequence[PassMeta]) -> Path:
    """Read the MAT file once and write the store to ``root`` (atomically, via a temp dir)."""
    mat_path = Path(raw) / MAT_FILE
    forces = dict(parse_datastore(read_datastore(mat_path), metas))
    source = {
        "source": str(mat_path),
        "source_bytes": mat_path.stat().st_size,
        "source_sha256": sha256_file(mat_path),
    }
    return write_force_store(root, forces, source)


def write_force_store(root: str | Path, forces: dict[str, np.ndarray], source: dict) -> Path:
    """Write ``forces`` ((samples, 3) arrays by record id) as a store. ``forces`` is consumed."""
    out = Path(root)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    groups = duplicate_groups(forces)
    representative = {record_id: record_id for record_id in forces}
    for group in groups:
        for member in group:
            representative[member] = group[0]
    distinct = sorted({rep for rep in representative.values()}, key=pass_order)
    total = sum(len(forces[record_id]) for record_id in distinct)
    signals = np.lib.format.open_memmap(
        tmp / "signals.npy", mode="w+", dtype=np.float32, shape=(total, 3)
    )
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for record_id in distinct:
        force = forces.pop(record_id)
        signals[offset : offset + len(force)] = force
        offsets[record_id] = (offset, len(force))
        offset += len(force)
    signals.flush()
    del signals, forces

    with (tmp / "index.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["record_id", "representative", "offset", "samples"])
        for record_id in sorted(representative, key=pass_order):
            rep = representative[record_id]
            writer.writerow([record_id, rep, *offsets[rep]])
    (tmp / "groups.json").write_text(json.dumps([list(group) for group in groups], indent=1))
    meta = {
        "store_version": STORE_VERSION,
        **source,
        "records": len(representative),
        "distinct_recordings": len(distinct),
        "duplicate_groups": len(groups),
        "total_samples": int(total),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out
