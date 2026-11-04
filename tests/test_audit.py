"""Audit checks: Nonastreda gap jumps (§2.5, P3) and the generic stage-step leakage probe
(§6.2 step 4, P9)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import meta

from rulfuse import paths
from rulfuse.adapters.nonastreda.audit import gap_wear_jumps, stage_steps_by_record
from rulfuse.adapters.nonastreda.load import metas_by_tool, photo_path
from rulfuse.audit import mean_luminance, stage_steps


def _linear_tool(tool: str, runs: list[int], stage_runs: tuple[int, int] = (4, 7)) -> list:
    """Wear rising 8 µm per run (2 per block), cut through every run 1..max(runs)."""
    out = []
    for run in runs:
        stage = 1 if run < stage_runs[0] else 2 if run < stage_runs[1] else 3
        for block in range(1, 5):
            out.append(
                meta(tool, run, block, tool_label=stage, wear=30.0 + 8 * (run - 1) + 2 * block)
            )
    return out


def test_gap_jump_ratio_counts_the_missing_runs():
    metas = _linear_tool("T1", [1, 2, 3, 4, 8, 9, 10])  # runs 5-7 cut but unrecorded
    rows = [r for r in gap_wear_jumps(metas) if r["kind"] == "interior"]
    assert len(rows) == 1
    assert rows[0]["missing_runs"] == [5, 6, 7]
    assert rows[0]["ratio_global"] == pytest.approx(4.0)  # 3 missing + 1
    assert rows[0]["expected_if_cut"] == 4


def test_stage_steps_separate_labels_from_wear():
    metas = [m for tool in ("T1", "T2", "T3") for m in _linear_tool(tool, list(range(1, 11)))]
    units = [m.tool for m in metas]
    groups = [f"R{m.run}" for m in metas]
    stages = [m.tool_label for m in metas]
    trend = np.array([m.flank_wear for m in metas])  # follows wear
    step = np.array([0.3 if m.tool_label == 1 else 0.5 for m in metas])  # follows the label
    result = stage_steps(units, groups, stages, {"trend": trend, "step": step})
    assert result["trend"]["largest_at_stage_boundary"] == 0 and not result["trend"]["leaks"]
    assert result["step"]["largest_at_stage_boundary"] == 3
    assert result["step"]["largest_by_transition"] == {"1to2": 3, "2to3": 0}
    assert result["step"]["median_share"]["1to2"] == pytest.approx(1.0)
    with_nan = trend.copy()
    with_nan[:4] = np.nan  # a group with no finite value is skipped, not an error
    assert stage_steps(units, groups, stages, {"x": with_nan})["x"]["units"] == 3


def test_leak_rule_counts_units():
    rng = np.random.default_rng(0)
    units, groups, stages, values = [], [], [], []
    for u in range(10):
        for g in range(8):
            stage = 1 if g < 4 else 2
            units.append(f"U{u}")
            groups.append(f"g{g}")
            stages.append(stage)
            jump = 5.0 if (stage == 2 and u < 4) else 0.0  # 4 units step at the boundary
            values.append(g * 0.1 + jump + rng.normal(0, 0.01))
    result = stage_steps(units, groups, stages, {"v": np.array(values)})["v"]
    assert result["largest_at_stage_boundary"] == 4 and result["leaks"]  # rule: > 3 units


@pytest.mark.data
def test_real_gaps_are_unrecorded_cutting(metas):
    rows = {r["tool"]: r for r in gap_wear_jumps(metas) if r["kind"] == "interior"}
    assert set(rows) == {"T2", "T4"}
    assert rows["T4"]["missing_runs"] == [6, 7, 8, 9]
    assert 4.5 <= rows["T4"]["ratio_global"] <= 6.5  # expected 5 if cut, 1 if not
    assert 1.5 <= rows["T2"]["ratio_local"] <= 3.0  # expected 2 if cut, 1 if not


@pytest.mark.data
def test_workpiece_brightness_encodes_the_stage_label(metas):
    """Documents the Stage-1 leakage finding (EDA_RUL.md §6): the workpiece photo's mean
    brightness jumps at the first run of tool_label stage 2 in every tool."""
    values = {
        m.record_id: {"work": mean_luminance(photo_path(paths.raw_dir(), m.record_id, "work"))}
        for m in metas
    }
    result = stage_steps_by_record(metas, values, ["work"])["work"]
    assert result["units"] == len(metas_by_tool(metas))
    assert result["largest_by_transition"]["1to2"] >= 9 and result["median_share"]["1to2"] >= 0.8
    assert result["leaks"]
