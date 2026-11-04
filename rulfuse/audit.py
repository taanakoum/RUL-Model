"""Data-audit checks of Stage 1 (PROJECT_PLAN.md §2.5, §17.2), reusable as leakage probes.

* ``gap_wear_jumps``: is a run gap unrecorded cutting? If the tool cut through k missing
  runs, the mean wear rises across the gap by about k + 1 run-to-run steps; if it did not,
  by about one step.
* ``stage_steps``: does a statistic encode the ``tool_label`` *stage* rather than wear?
  ``tool_label`` changes only at a run's first block. A statistic that follows wear rises
  over many run boundaries; one tied to the stage label (for example an acquisition change
  made when the stage changed) jumps at the stage boundary and nowhere else.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rulfuse.data.nonastreda import PassMeta, metas_by_tool
from rulfuse.data.timeline import run_gaps


def gap_wear_jumps(metas: Sequence[PassMeta], local_runs: int = 3) -> list[dict]:
    """One row per interior run gap and per leading gap, for every tool."""
    by_tool = metas_by_tool(metas)
    first_run_wear = {
        tool: float(np.mean([m.flank_wear for m in ms if m.run == min(x.run for x in ms)]))
        for tool, ms in by_tool.items()
    }
    rows: list[dict] = []
    for tool, tool_metas in by_tool.items():
        wear_by_run: dict[int, list[float]] = defaultdict(list)
        for meta in tool_metas:
            wear_by_run[meta.run].append(meta.flank_wear)
        runs = sorted(wear_by_run)
        means = {run: float(np.mean(wear_by_run[run])) for run in runs}
        steps = [(a, b, means[b] - means[a]) for a, b in zip(runs, runs[1:], strict=False)]
        consecutive = [delta for a, b, delta in steps if b - a == 1]
        for a, b, delta in steps:
            if b - a == 1:
                continue
            local = [
                d
                for x, y, d in steps
                if y - x == 1 and (abs(x - a) <= local_runs or abs(y - b) <= local_runs)
            ]
            global_step = float(np.median(consecutive))
            local_step = float(np.median(local)) if local else global_step
            rows.append(
                {
                    "tool": tool,
                    "kind": "interior",
                    "missing_runs": list(range(a + 1, b)),
                    "wear_before_um": round(means[a], 1),
                    "wear_after_um": round(means[b], 1),
                    "jump_um": round(delta, 1),
                    "median_step_um": round(global_step, 1),
                    "local_median_step_um": round(local_step, 1),
                    "ratio_global": round(delta / global_step, 2),
                    "ratio_local": round(delta / local_step, 2),
                    "expected_if_cut": b - a,
                    "expected_if_not_cut": 1,
                }
            )
        gaps = run_gaps(tool_metas)
        if gaps.leading:
            others = [value for other, value in first_run_wear.items() if other != tool]
            rows.append(
                {
                    "tool": tool,
                    "kind": "leading",
                    "missing_runs": list(gaps.leading),
                    "first_recorded_run_wear_um": round(first_run_wear[tool], 1),
                    "other_tools_first_run_wear_um": [round(min(others), 1), round(max(others), 1)],
                    "other_tools_first_run_median_um": round(float(np.median(others)), 1),
                }
            )
    return rows


def stage_steps(
    metas: Sequence[PassMeta],
    values: Mapping[str, Mapping[str, float]],
    statistics: Sequence[str],
) -> dict[str, dict]:
    """For each statistic: in how many tools the stage-1→2 (2→3) run boundary is the largest
    of all run-boundary steps, and the median share of the tool's range that step covers.

    ``values[record_id][statistic]`` holds the per-pass values (missing or NaN are skipped).
    """
    by_tool = metas_by_tool(metas)
    out: dict[str, dict] = {}
    for statistic in statistics:
        largest = {"1to2": 0, "2to3": 0}
        shares: dict[str, list[float]] = {"1to2": [], "2to3": []}
        tools_used = 0
        for tool_metas in by_tool.values():
            per_run: dict[int, list[float]] = defaultdict(list)
            stage: dict[int, int] = {}
            for meta in tool_metas:
                value = values.get(meta.record_id, {}).get(statistic)
                if value is not None and np.isfinite(value):
                    per_run[meta.run].append(float(value))
                stage[meta.run] = max(stage.get(meta.run, 0), meta.tool_label)
            ordered = [run for run in sorted(stage) if per_run.get(run)]
            if len(ordered) < 3:
                continue
            tools_used += 1
            run_mean = {run: float(np.mean(per_run[run])) for run in ordered}
            steps = {
                b: run_mean[b] - run_mean[a] for a, b in zip(ordered, ordered[1:], strict=False)
            }
            span = max(run_mean.values()) - min(run_mean.values())
            biggest = max(steps, key=lambda run: abs(steps[run]))
            for name, (lo, hi) in (("1to2", (1, 2)), ("2to3", (2, 3))):
                boundary = next(
                    (
                        b
                        for a, b in zip(ordered, ordered[1:], strict=False)
                        if stage[a] == lo and stage[b] == hi
                    ),
                    None,
                )
                if boundary is None:
                    continue
                largest[name] += int(biggest == boundary)
                shares[name].append(abs(steps[boundary]) / span if span > 0 else 0.0)
        out[statistic] = {
            "tools": tools_used,
            "largest_at_1to2": largest["1to2"],
            "largest_at_2to3": largest["2to3"],
            "share_1to2": round(float(np.median(shares["1to2"])), 2) if shares["1to2"] else None,
            "share_2to3": round(float(np.median(shares["2to3"])), 2) if shares["2to3"] else None,
        }
    return out


def mean_luminance(path: str | Path) -> float:
    """Mean grey level (0-1) of an image, read at reduced size."""
    from PIL import Image

    with Image.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", (image.width // 4, image.height // 4))
        grey = image.convert("L")
    grey.thumbnail((256, 256))
    return float(np.asarray(grey, dtype=np.float64).mean() / 255.0)
