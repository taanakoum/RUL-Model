"""The physical timeline of each tool (PROJECT_PLAN.md D8, D12, §2.4, §2.5, §5.3).

A timeline is the ordered list of the passes a tool actually cut, each with a cutting
duration and its cumulative cutting time.

Axes (D12):

* ``"gap"`` (default): positions follow (run, block). Runs missing *between* a tool's first
  and last recorded run are inserted as 4 unrecorded passes each. They carry no data and no
  prediction points. Their duration is the median of the tool's recorded passes *before*
  the gap, so the cumulative cutting time after a gap uses only the past (D14). Runs
  missing *before* the first recorded run are inserted only with ``count_leading=True``,
  at ``DEFAULT_PASS_SECONDS`` (no earlier pass of the tool exists).
* ``"recorded"``: only the recorded passes, numbered consecutively.

Duplicated force recordings (D8):

* ``"latest"`` (default): the latest member of a group carries the recording; the earlier
  members are force-missing. They keep their photos, labels and duration (the copy's).
* ``"every"``: every member carries the recording (may show the future by one pass).
* ``"dedup"``: OmniFuser's deduplication. One member per group is kept (the largest flank
  wear, then the earliest block) and the others are dropped from the timeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rulfuse.adapters.nonastreda.load import BLOCKS_PER_RUN, PassMeta, metas_by_tool

AXES = ("gap", "recorded")
ATTACH_RULES = ("latest", "every", "dedup")
# Median trimmed pass duration over all 512 passes (OmniFuser EDA: 98.5 s raw, 97 s trimmed).
# Used only for leading gaps, which have no earlier pass of the same tool.
DEFAULT_PASS_SECONDS = 97.0


@dataclass(frozen=True)
class Slot:
    """One physical pass of a tool."""

    tool: str
    position: int  # 1-based position on the chosen axis
    run: int
    block: int
    record_id: str | None  # None: an unrecorded pass (a gap)
    force_record: str | None  # the recording this pass carries; None: force-missing
    duration_s: float
    tau_start_s: float  # cumulative cutting time at the start of the pass

    @property
    def tau_end_s(self) -> float:
        return self.tau_start_s + self.duration_s

    @property
    def recorded(self) -> bool:
        return self.record_id is not None

    @property
    def force_present(self) -> bool:
        return self.force_record is not None


@dataclass(frozen=True)
class RunGaps:
    tool: str
    first_run: int
    last_run: int
    leading: tuple[int, ...]  # missing runs before the first recorded run
    interior: tuple[int, ...]  # missing runs between the first and last recorded run


def run_gaps(metas: Sequence[PassMeta]) -> RunGaps:
    """Missing run numbers of one tool."""
    runs = sorted({meta.run for meta in metas})
    if not runs:
        raise ValueError("a tool needs at least one recorded pass")
    present = set(runs)
    return RunGaps(
        tool=metas[0].tool,
        first_run=runs[0],
        last_run=runs[-1],
        leading=tuple(range(1, runs[0])),
        interior=tuple(run for run in range(runs[0], runs[-1] + 1) if run not in present),
    )


def _force_records(
    metas: Sequence[PassMeta], groups: Sequence[tuple[str, ...]], attach: str
) -> tuple[dict[str, str | None], set[str]]:
    """(record id -> the recording it carries or None, record ids dropped by deduplication)."""
    ids = {meta.record_id for meta in metas}
    by_id = {meta.record_id: meta for meta in metas}
    carried: dict[str, str | None] = {record_id: record_id for record_id in ids}
    dropped: set[str] = set()
    for group in groups:
        if not set(group) <= ids:
            continue
        if attach == "latest":
            for member in group[:-1]:
                carried[member] = None
        elif attach == "dedup":
            keep = max(group, key=lambda r: (by_id[r].flank_wear, -by_id[r].block))
            dropped.update(member for member in group if member != keep)
    return carried, dropped


def build_tool_timeline(
    metas: Sequence[PassMeta],
    durations: Mapping[str, float],
    groups: Sequence[tuple[str, ...]] = (),
    *,
    axis: str = "gap",
    count_leading: bool = False,
    attach: str = "latest",
) -> list[Slot]:
    """The timeline of one tool. ``durations`` gives every recorded pass's trimmed seconds."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}")
    if attach not in ATTACH_RULES:
        raise ValueError(f"attach must be one of {ATTACH_RULES}")
    ordered = sorted(metas, key=lambda meta: (meta.run, meta.block))
    if len({meta.tool for meta in ordered}) != 1:
        raise ValueError("build_tool_timeline takes the passes of exactly one tool")
    carried, dropped = _force_records(ordered, groups, attach)
    kept = [meta for meta in ordered if meta.record_id not in dropped]
    missing = [meta.record_id for meta in kept if meta.record_id not in durations]
    if missing:
        raise ValueError(f"no duration for {missing[:3]}")
    tool = ordered[0].tool

    def gap_duration(run: int) -> float:
        before = [durations[meta.record_id] for meta in kept if meta.run < run]
        return float(np.median(before)) if before else DEFAULT_PASS_SECONDS

    entries: list[tuple[int, int, PassMeta | None]] = []
    if axis == "recorded":
        entries = [(meta.run, meta.block, meta) for meta in kept]
    else:
        gaps = run_gaps(ordered)
        by_run: dict[int, list[PassMeta]] = {}
        for meta in kept:
            by_run.setdefault(meta.run, []).append(meta)
        first = 1 if count_leading else gaps.first_run
        unrecorded_runs = set(gaps.interior) | (set(gaps.leading) if count_leading else set())
        for run in range(first, gaps.last_run + 1):
            if run in unrecorded_runs:
                entries.extend((run, block, None) for block in range(1, BLOCKS_PER_RUN + 1))
            else:
                entries.extend((meta.run, meta.block, meta) for meta in by_run.get(run, []))

    slots: list[Slot] = []
    tau = 0.0
    for position, (run, block, meta) in enumerate(entries, start=1):
        duration = gap_duration(run) if meta is None else float(durations[meta.record_id])
        slots.append(
            Slot(
                tool=tool,
                position=position,
                run=run,
                block=block,
                record_id=None if meta is None else meta.record_id,
                force_record=None if meta is None else carried[meta.record_id],
                duration_s=duration,
                tau_start_s=tau,
            )
        )
        tau += duration
    return slots


def build_timelines(
    metas: Sequence[PassMeta],
    durations: Mapping[str, float],
    groups: Sequence[tuple[str, ...]] = (),
    **options: object,
) -> dict[str, list[Slot]]:
    """Timelines of every tool, in tool order (options as in ``build_tool_timeline``)."""
    return {
        tool: build_tool_timeline(tool_metas, durations, groups, **options)  # type: ignore[arg-type]
        for tool, tool_metas in metas_by_tool(metas).items()
    }
