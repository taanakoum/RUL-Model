"""Nonastreda raw data: labels, photo paths, force recordings and duplicate groups.

Vendored in part from OmniFuser (``omnifuser/data.py``, commit e3b9985, branch
``Ruche-OmniFuser-Training``): ``RECORD_ID``, ``read_csv`` (OmniFuser ``_read_csv``),
``decode_record_id`` (``_decode_record_id``) and the checks of ``load_nonastreda`` (label
agreement between the MAT file and labels.csv, the ``(3, samples)`` shape, finite values).

Unlike OmniFuser, all 512 recorded passes are kept. Bit-identical force recordings inside a
run are returned as duplicate groups instead of being merged (PROJECT_PLAN.md §2.4, D8).
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLASS_NAMES = ("sharp", "used", "dulled")
RECORD_ID = re.compile(r"^T(?P<tool>\d+)R(?P<run>\d+)B(?P<block>\d+)$")
PHOTO_KINDS: dict[str, str] = {"tool": ".jpg", "chip": ".jpg", "work": ".png"}
BLOCKS_PER_RUN = 4
SAMPLE_RATE_HZ = 1000
FORCE_CHANNELS = ("Fx", "Fy", "Fz")
MAT_FILE = "forces_xyz_raw.mat"


@dataclass(frozen=True)
class PassMeta:
    """One recorded pass: its identity and its labels (never a model input)."""

    record_id: str
    tool: str
    run: int
    block: int
    image_label: str
    tool_label: int
    flank_wear: float
    gaps: float
    overhang: float


def tool_number(tool: str) -> int:
    return int(tool[1:])


def pass_order(record_id: str) -> tuple[int, int, int]:
    """Sort key: tool number, run, block."""
    tool, run, block, _ = decode_record_id(record_id)
    return tool_number(tool), run, block


def read_csv(path: str | Path) -> dict[str, dict[str, str]]:
    """Rows of a CSV with an ``id`` column, keyed by id (vendored from OmniFuser)."""
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows or "id" not in rows[0]:
        raise ValueError(f"{path} must have a header and an id column")
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate record ids")
    return result


def decode_record_id(name: str) -> tuple[str, int, int, str]:
    """``T10R3B4`` (or a file name such as ``T10R3B4.jpg``) -> ("T10", 3, 4, "T10R3B4")."""
    record_id = Path(name).stem
    match = RECORD_ID.fullmatch(record_id)
    if match is None:
        raise ValueError(f"unexpected Nonastreda record id: {record_id!r}")
    return (
        f"T{int(match.group('tool'))}",
        int(match.group("run")),
        int(match.group("block")),
        record_id,
    )


def load_labels(raw: str | Path) -> list[PassMeta]:
    """Every recorded pass with its labels, in (tool, run, block) order, validated."""
    root = Path(raw)
    labels = read_csv(root / "labels.csv")
    regression = read_csv(root / "labels_reg.csv")
    if set(labels) != set(regression):
        missing, extra = set(labels) - set(regression), set(regression) - set(labels)
        raise ValueError(
            f"labels_reg.csv ids must match labels.csv (missing={len(missing)}, extra={len(extra)})"
        )
    metas: list[PassMeta] = []
    for record_id, row in labels.items():
        tool, run, block, record_id = decode_record_id(record_id)
        if not 1 <= block <= BLOCKS_PER_RUN:
            raise ValueError(f"{record_id}: block must be 1..{BLOCKS_PER_RUN}")
        image_label = row.get("image_label", "").strip().lower()
        if image_label not in CLASS_NAMES:
            raise ValueError(f"unknown image label {image_label!r} for {record_id}")
        try:
            tool_label = int(row.get("tool_label", ""))
        except ValueError as error:
            raise ValueError(f"invalid tool label for {record_id}") from error
        if tool_label not in (1, 2, 3):
            raise ValueError(f"tool label for {record_id} must be 1, 2 or 3")
        values: dict[str, float] = {}
        for key in ("flank_wear", "gaps", "overhang"):
            try:
                value = float(regression[record_id][key])
            except (KeyError, ValueError) as error:
                raise ValueError(f"invalid {key} for {record_id}") from error
            if not np.isfinite(value):
                raise ValueError(f"non-finite {key} for {record_id}")
            values[key] = value
        metas.append(PassMeta(record_id, tool, run, block, image_label, tool_label, **values))
    return sorted(metas, key=lambda meta: pass_order(meta.record_id))


def metas_by_tool(metas: Sequence[PassMeta]) -> dict[str, list[PassMeta]]:
    """Recorded passes per tool, in (run, block) order; tools in numeric order."""
    grouped: dict[str, list[PassMeta]] = defaultdict(list)
    for meta in sorted(metas, key=lambda item: pass_order(item.record_id)):
        grouped[meta.tool].append(meta)
    return dict(sorted(grouped.items(), key=lambda item: tool_number(item[0])))


def photo_path(raw: str | Path, record_id: str, kind: str) -> Path:
    if kind not in PHOTO_KINDS:
        raise ValueError(f"photo kind must be one of {sorted(PHOTO_KINDS)}")
    return Path(raw) / kind / f"{record_id}{PHOTO_KINDS[kind]}"


def read_datastore(mat_path: str | Path) -> np.ndarray:
    """The MAT file's ``baseDatastore`` table: rows of (file name, label, ..., force)."""
    from scipy.io import loadmat

    contents = loadmat(str(mat_path), variable_names=["baseDatastore"], simplify_cells=True)
    datastore = contents.get("baseDatastore")
    if not isinstance(datastore, np.ndarray) or datastore.ndim != 2 or datastore.shape[1] < 4:
        raise ValueError(f"{mat_path}: baseDatastore must be a table with file name, label, force")
    return datastore


def parse_datastore(
    datastore: np.ndarray, metas: Sequence[PassMeta]
) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (record id, force as (samples, 3) float32) for every row, with OmniFuser's checks.

    Each row's MATLAB copy is released as it is converted, so the peak memory stays near one
    float64 copy of the data.
    """
    by_id = {meta.record_id: meta for meta in metas}
    seen: set[str] = set()
    for row_index, row in enumerate(datastore):
        _, _, _, record_id = decode_record_id(str(row[0]))
        if record_id in seen:
            raise ValueError(f"duplicate force record {record_id}")
        seen.add(record_id)
        if record_id not in by_id:
            raise ValueError(f"{record_id} is missing from labels.csv")
        mat_label = str(row[1]).strip().lower()
        if mat_label != by_id[record_id].image_label:
            raise ValueError(
                f"label mismatch for {record_id}: MAT={mat_label}, "
                f"CSV={by_id[record_id].image_label}"
            )
        raw_force = np.asarray(row[3])
        if raw_force.ndim != 2 or raw_force.shape[0] != len(FORCE_CHANNELS):
            raise ValueError(
                f"{record_id}: force must have shape (3, samples), got {raw_force.shape}"
            )
        if not np.isfinite(raw_force).all():
            raise ValueError(f"{record_id}: force contains non-finite values")
        force = np.ascontiguousarray(raw_force.T, dtype=np.float32)
        datastore[row_index, 3] = None
        yield record_id, force
    missing = set(by_id) - seen
    if missing:
        raise ValueError(f"labels.csv has {len(missing)} id(s) without force records")


def duplicate_groups(forces: Mapping[str, np.ndarray]) -> list[tuple[str, ...]]:
    """Groups (two or more) of bit-identical recordings within one (tool, run).

    Members are in block order; groups in (tool, run, first block) order. Hashes find the
    candidates and ``np.array_equal`` confirms them, as OmniFuser's exact comparison does.
    """
    buckets: dict[tuple[str, int, tuple[int, ...], str], list[tuple[int, str]]] = defaultdict(list)
    for record_id, force in forces.items():
        tool, run, block, _ = decode_record_id(record_id)
        digest = hashlib.sha1(np.ascontiguousarray(force).tobytes()).hexdigest()
        buckets[(tool, run, tuple(force.shape), digest)].append((block, record_id))
    groups: list[tuple[str, ...]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        ids = tuple(record_id for _, record_id in sorted(members))
        reference = forces[ids[0]]
        if not all(np.array_equal(reference, forces[other]) for other in ids[1:]):
            raise RuntimeError(f"hash collision without equal arrays in {ids}")
        groups.append(ids)
    return sorted(groups, key=lambda group: pass_order(group[0]))
