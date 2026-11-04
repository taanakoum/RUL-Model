"""Stage-1 audit checks (PROJECT_PLAN.md §2.5, P3 scientific validation; the stage-step
leakage probe used again on encoder features in Gate 1)."""

from __future__ import annotations

import pytest
from helpers import meta

from rulfuse import paths
from rulfuse.audit import gap_wear_jumps, mean_luminance, stage_steps
from rulfuse.data.nonastreda import metas_by_tool, photo_path


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
    trend = {m.record_id: {"stat": m.flank_wear} for m in metas}  # follows wear
    step = {m.record_id: {"stat": 0.3 if m.tool_label == 1 else 0.5} for m in metas}  # the label
    assert stage_steps(metas, trend, ["stat"])["stat"]["largest_at_1to2"] == 0
    result = stage_steps(metas, step, ["stat"])["stat"]
    assert result["largest_at_1to2"] == 3 and result["share_1to2"] == pytest.approx(1.0)


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
    result = stage_steps(metas, values, ["work"])["work"]
    assert result["tools"] == len(metas_by_tool(metas))
    assert result["largest_at_1to2"] >= 9 and result["share_1to2"] >= 0.8
