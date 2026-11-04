"""Builders of small synthetic inputs for the tests."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from rulfuse.data.nonastreda import PassMeta


def meta(
    tool: str = "T1",
    run: int = 1,
    block: int = 1,
    label: str = "sharp",
    tool_label: int = 1,
    wear: float = 50.0,
) -> PassMeta:
    return PassMeta(f"{tool}R{run}B{block}", tool, run, block, label, tool_label, wear, 0.0, 0.0)


def tool_metas(
    labels: Iterable[str],
    tool: str = "T1",
    runs: Iterable[int] | None = None,
    wear_step: float = 5.0,
) -> list[PassMeta]:
    """Passes of one tool, 4 per run, with the given labels in order."""
    labels = list(labels)
    run_numbers = list(runs) if runs is not None else list(range(1, len(labels) // 4 + 2))
    out: list[PassMeta] = []
    slots = [(run, block) for run in run_numbers for block in range(1, 5)]
    for index, label in enumerate(labels):
        run, block = slots[index]
        out.append(meta(tool, run, block, label, 1, 40.0 + wear_step * index))
    return out


def write_label_csvs(root: Path, rows: list[dict[str, str]]) -> None:
    """labels.csv (id, image_label, tool_label) and labels_reg.csv (id, gaps, flank_wear,
    overhang)."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["id", "image_label", "tool_label"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ("id", "image_label", "tool_label")})
    with (root / "labels_reg.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, ["id", "gaps", "flank_wear", "overhang"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ("id", "gaps", "flank_wear", "overhang")})


def label_row(
    record_id: str, label: str = "sharp", tool_label: str = "1", wear: str = "50"
) -> dict:
    return {
        "id": record_id,
        "image_label": label,
        "tool_label": tool_label,
        "gaps": "0",
        "flank_wear": wear,
        "overhang": "0",
    }


SYNTHETIC_PLAN = {
    # tool: (recorded runs, labels in pass order)
    "T1": ([1, 2, 3], ["sharp"] * 6 + ["used"] * 3 + ["dulled"] * 3),
    "T2": ([1, 2, 4], ["sharp"] * 4 + ["used"] * 2 + ["dulled"] * 6),  # run 3 unrecorded
    "T3": ([1, 2, 3], ["sharp"] * 5 + ["used"] * 3 + ["dulled"] * 4),
}
STAGE = {"sharp": "1", "used": "2", "dulled": "3"}


def synthetic_dataset_files(root: Path) -> tuple[Path, Path]:
    """Raw label files and a force store for 3 small tools.

    Passes are 0.8 s idle, 23-25 s of cutting, 0.8 s idle (so the trim removes the idle
    ends and leaves 2 or 3 segments). T1R2B4 is a bit-identical copy of T1R2B3.
    """
    import numpy as np

    from rulfuse.data.force_store import write_force_store

    raw, store = root / "raw", root / "forces"
    rng = np.random.default_rng(0)
    rows, forces = [], {}
    for tool, (runs, labels) in SYNTHETIC_PLAN.items():
        slots = [(run, block) for run in runs for block in range(1, 5)]
        for index, ((run, block), label) in enumerate(zip(slots, labels, strict=True)):
            record_id = f"{tool}R{run}B{block}"
            rows.append(label_row(record_id, label, STAGE[label], str(40 + 8 * index)))
            cut = int((23 + index % 3) * 1000)
            force = np.zeros((cut + 1600, 3), np.float32)
            force[:, 0] = 5.0
            force[800 : 800 + cut, 0] = 100 + 2 * index + rng.normal(0, 2, cut)
            force[800 : 800 + cut, 2] = -10 - 0.01 * np.arange(cut) / 100  # a little drift
            forces[record_id] = force
    forces["T1R2B4"] = forces["T1R2B3"].copy()
    write_label_csvs(raw, rows)
    write_force_store(
        store, forces, {"source": "synthetic", "source_bytes": 0, "source_sha256": "synthetic"}
    )
    return raw, store
