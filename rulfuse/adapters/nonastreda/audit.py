"""Nonastreda data-audit checks of Stage 1 (PROJECT_PLAN.md §2.5).

* ``gap_wear_jumps``: is a run gap unrecorded cutting? If the tool cut through k missing
  runs, the mean wear rises across the gap by about k + 1 run-to-run steps; if it did not,
  by about one step.
* ``stage_steps_by_record``: the generic :func:`rulfuse.audit.stage_steps` on per-pass
  values, with runs as groups and ``tool_label`` as the stage.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from rulfuse.adapters.nonastreda.load import PassMeta, metas_by_tool
from rulfuse.adapters.nonastreda.timeline import run_gaps
from rulfuse.audit import stage_steps


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


def stage_steps_by_record(
    metas: Sequence[PassMeta],
    values: Mapping[str, Mapping[str, float]],
    statistics: Sequence[str],
) -> dict[str, dict]:
    """``values[record_id][statistic]``: per-pass values (missing or NaN are skipped)."""
    ordered = [m for tool_metas in metas_by_tool(metas).values() for m in tool_metas]
    columns = {
        name: np.array([values.get(m.record_id, {}).get(name, np.nan) for m in ordered], float)
        for name in statistics
    }
    return stage_steps(
        [m.tool for m in ordered],
        [f"R{m.run}" for m in ordered],
        [m.tool_label for m in ordered],
        columns,
    )
