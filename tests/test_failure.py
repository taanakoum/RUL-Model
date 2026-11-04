"""P5 (adapter part): Nonastreda TTF rules, isotonic wear and failure times (PROJECT_PLAN.md
D2, §2.3, §5.6)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import meta, tool_metas

from rulfuse.adapters.nonastreda.failure import isotonic, isotonic_causal, tool_failure, ttf_index
from rulfuse.adapters.nonastreda.timeline import build_tool_timeline

SEQ = ["sharp", "sharp", "dulled", "used", "sharp", "dulled", "dulled", "used", "dulled"]


def test_ttf_rules_on_a_hand_made_sequence():
    metas = tool_metas(SEQ)
    assert ttf_index("literal", metas) == 2
    assert ttf_index("confirmed", metas) == 5  # d, then d, u, d: 2 of the next 3
    assert ttf_index("dulled_onwards", metas) == 8
    assert ttf_index("confirmed", tool_metas(["dulled"])) is None
    assert ttf_index("confirmed", tool_metas(["dulled", "dulled", "dulled"])) == 0
    assert ttf_index("dulled_onwards", tool_metas(["sharp", "used"])) is None
    assert ttf_index("dulled_onwards", tool_metas(["dulled", "dulled"])) == 0
    with pytest.raises(ValueError):
        ttf_index("unknown", metas)


def test_wear_and_tool_label_rules():
    wears = [60, 100, 112, 105, 108, 115]  # isotonic: 60, 100, 108.33 x3, 115
    metas = [meta("T1", 1 + i // 4, 1 + i % 4, wear=w) for i, w in enumerate(wears)]
    assert ttf_index("wear110", metas) == 5
    labels = [
        meta("T1", 1, b, tool_label=t) for b, t in zip(range(1, 5), (1, 2, 3, 3), strict=True)
    ]
    assert ttf_index("tool_label3", labels) == 2


def test_isotonic_and_its_causal_version():
    values = [68, 72, 68, 76]
    np.testing.assert_allclose(isotonic(values), [68, 70, 70, 76])
    causal = isotonic_causal(values)
    np.testing.assert_allclose(causal, [68, 72, 70, 76])
    changed = isotonic_causal([68, 72, 68, 10])  # a later value never changes earlier ones
    np.testing.assert_allclose(changed[:3], causal[:3])
    assert np.all(np.diff(isotonic(np.random.default_rng(0).normal(size=50))) >= -1e-12)


def test_failure_time_is_the_end_of_the_ttf_pass_on_the_gap_axis():
    """Runs 1 and 3 (run 2 a gap of 4 unrecorded passes), 100 s passes; TTF at R3B2."""
    labels = ["sharp"] * 4 + ["sharp", "dulled", "dulled", "dulled"]
    metas = tool_metas(labels, runs=[1, 3])
    timeline = build_tool_timeline(metas, {m.record_id: 100.0 for m in metas})
    failure = tool_failure(timeline, metas)
    assert failure.ttf_record == "T1R3B2" and failure.ttf_position == 10
    assert failure.tau_ttf == pytest.approx(1000.0)  # 10 passes of 100 s, gaps included
    with pytest.raises(ValueError, match="censored"):
        tool_failure(timeline, tool_metas(["sharp"] * 8, runs=[1, 3]))


@pytest.mark.data
def test_real_ttf_tables(data, data_recorded):
    gap = [data.failures[t].ttf_position for t in data.tools]
    recorded = [data_recorded.failures[t].ttf_position for t in data_recorded.tools]
    assert gap == [31, 44, 36, 38, 19, 47, 47, 46, 44, 44]  # PROJECT_PLAN.md §2.3
    assert recorded == [31, 40, 36, 22, 19, 47, 47, 46, 44, 44]
    wear = [data.failures[t].wear_at_ttf_um for t in data.tools]
    assert 108.5 <= min(wear) and max(wear) <= 123.0  # smoothed wear at the TTF (§2.2)


@pytest.mark.data
@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("literal", [31, 44, 20, 38, 16, 24, 47, 27, 44, 42]),
        ("wear110", [31, 44, 28, 38, 23, 50, 49, 45, 44, 44]),
        ("tool_label3", [29, 45, 25, 37, 21, 49, 49, 45, 45, 45]),
        ("dulled_onwards", [31, 48, 39, 38, 31, 54, 49, 50, 44, 46]),
    ],
)
def test_real_alternative_rules(data, rule, expected):
    from rulfuse.adapters.nonastreda.load import metas_by_tool

    by_tool = metas_by_tool(data.metas)
    got = [tool_failure(data.timelines[t], by_tool[t], rule).ttf_position for t in data.tools]
    assert got == expected
