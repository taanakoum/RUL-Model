"""P3 Timeline: gap-aware axis and duplicate attachment (PROJECT_PLAN.md D8, D12, §15)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from helpers import meta

from rulfuse.data.timeline import DEFAULT_PASS_SECONDS, build_tool_timeline, run_gaps


def tool_with_gaps():
    """Runs 2, 3 and 5 recorded (run 1 leading gap, run 4 interior gap); R3B3 = R3B4."""
    metas = [
        meta("T1", run, block, wear=10.0 * run + block)
        for run in (2, 3, 5)
        for block in range(1, 5)
    ]
    durations = {m.record_id: 90.0 + m.run for m in metas}
    groups = [("T1R3B3", "T1R3B4")]
    return metas, durations, groups


def test_run_gaps():
    metas, _, _ = tool_with_gaps()
    gaps = run_gaps(metas)
    assert gaps.leading == (1,) and gaps.interior == (4,)


def test_gap_axis_inserts_interior_runs_with_past_median():
    metas, durations, groups = tool_with_gaps()
    slots = build_tool_timeline(metas, durations, groups)
    assert [s.position for s in slots] == list(range(1, 17))
    gap = [s for s in slots if not s.recorded]
    assert [(s.run, s.block) for s in gap] == [(4, b) for b in range(1, 5)]
    # Median of the recorded passes *before* the gap (runs 2 and 3: 92 and 93 s).
    assert all(s.duration_s == pytest.approx(92.5) for s in gap)
    assert all(not s.force_present for s in gap)
    taus = np.array([s.tau_start_s for s in slots])
    np.testing.assert_allclose(np.diff(taus), [s.duration_s for s in slots[:-1]])


def test_leading_gap_only_on_request():
    metas, durations, groups = tool_with_gaps()
    slots = build_tool_timeline(metas, durations, groups, count_leading=True)
    assert len(slots) == 20
    assert all(not s.recorded and s.duration_s == DEFAULT_PASS_SECONDS for s in slots[:4])


def test_recorded_axis_has_no_gaps():
    metas, durations, groups = tool_with_gaps()
    slots = build_tool_timeline(metas, durations, groups, axis="recorded")
    assert len(slots) == 12 and all(s.recorded for s in slots)


def test_duplicate_attachment_rules():
    metas, durations, groups = tool_with_gaps()
    latest = {s.record_id: s for s in build_tool_timeline(metas, durations, groups) if s.recorded}
    assert not latest["T1R3B3"].force_present and latest["T1R3B4"].force_present
    assert latest["T1R3B3"].duration_s == durations["T1R3B3"]
    every = {
        s.record_id: s
        for s in build_tool_timeline(metas, durations, groups, attach="every")
        if s.recorded
    }
    assert every["T1R3B3"].force_present and every["T1R3B4"].force_present
    dedup = build_tool_timeline(metas, durations, groups, attach="dedup")
    ids = [s.record_id for s in dedup if s.recorded]
    assert "T1R3B4" in ids and "T1R3B3" not in ids  # larger wear kept
    assert len([s for s in dedup if not s.recorded]) == 4  # a dropped block is not a gap


def test_rejects_mixed_tools_and_bad_options():
    metas, durations, groups = tool_with_gaps()
    with pytest.raises(ValueError):
        build_tool_timeline([*metas, meta("T2", 1, 1)], durations, groups)
    with pytest.raises(ValueError):
        build_tool_timeline(metas, durations, groups, axis="other")


@pytest.mark.data
def test_real_timelines(dataset):
    unrecorded = {
        tool: sum(not s.recorded for s in slots) for tool, slots in dataset.timelines.items()
    }
    assert unrecorded == {
        "T1": 0,
        "T2": 4,
        "T3": 0,
        "T4": 16,
        "T5": 0,
        "T6": 0,
        "T7": 0,
        "T8": 0,
        "T9": 0,
        "T10": 0,
    }
    missing = [
        s
        for slots in dataset.timelines.values()
        for s in slots
        if s.recorded and not s.force_present
    ]
    assert len(missing) == 134
    assert Counter(s.block for s in missing)[3] >= 127
