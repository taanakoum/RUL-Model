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
