"""P5 Labels: TTF rules, isotonic wear, points and RUL targets (PROJECT_PLAN.md §3, §15)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import meta, tool_metas

from rulfuse.data.labels import (
    isotonic,
    isotonic_causal,
    prediction_points,
    rul_targets,
    tool_labels,
    ttf_index,
)
from rulfuse.data.timeline import build_tool_timeline

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


def _simple_timeline():
    """Runs 1 and 3 (run 2 a gap), 100 s passes; TTF at R3B2 (position 10)."""
    labels = ["sharp"] * 4 + ["sharp", "dulled", "dulled", "dulled"]
    metas = tool_metas(labels, runs=[1, 3])
    durations = {m.record_id: 100.0 for m in metas}
    return metas, build_tool_timeline(metas, durations)


def test_points_skip_gaps_and_targets_follow_the_definitions():
    metas, timeline = _simple_timeline()
    labels = tool_labels(timeline, metas)
    assert labels.ttf_record == "T1R3B2" and labels.ttf_position == 10
    points = labels.points
    assert set(points.position) == {1, 2, 3, 4, 9, 10}  # gap passes 5-8 have no points
    assert len(points) == 60
    first = 0
    assert points.t_label[first] == pytest.approx(0.1) and points.tau[first] == pytest.approx(10.0)
    assert np.isnan(points.mean_past_duration[first])
    targets = labels.targets
    assert targets.rul_pass[first] == pytest.approx(9.9)  # TTF at the end of pass 10
    assert targets.rul_sec[first] == pytest.approx(990.0)
    assert targets.rul_pass[-1] == pytest.approx(0.0) and targets.rul_sec[-1] == pytest.approx(0.0)
    assert np.all(targets.y_pass <= 15) and np.all(targets.y_sec <= 1500)
    assert targets.window_pass.all()  # every point is within 15 passes of failure here


def test_targets_reject_points_after_the_ttf():
    metas, timeline = _simple_timeline()
    points = prediction_points(timeline, last_position=10)
    with pytest.raises(ValueError):
        rul_targets(points, ttf_position=9, tau_ttf=900.0)


@pytest.mark.data
def test_real_ttf_tables(dataset, dataset_recorded):
    gap = [dataset.labels[t].ttf_position for t in dataset.tools]
    recorded = [dataset_recorded.labels[t].ttf_position for t in dataset_recorded.tools]
    assert gap == [31, 44, 36, 38, 19, 47, 47, 46, 44, 44]  # PROJECT_PLAN.md §2.3
    assert recorded == [31, 40, 36, 22, 19, 47, 47, 46, 44, 44]
    wear = [dataset.labels[t].wear_at_ttf_um for t in dataset.tools]
    assert 108.5 <= min(wear) and max(wear) <= 123.0  # smoothed wear at the TTF (§2.2)
    windows = sum(int(item.targets.window_pass.sum()) for item in dataset.labels.values())
    windows_sec = sum(int(item.targets.window_sec.sum()) for item in dataset.labels.values())
    assert (windows, windows_sec) == (1338, 1361)
    assert int(dataset.labels["T4"].targets.window_pass.sum()) == 20


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
def test_real_alternative_rules(dataset, rule, expected):
    from rulfuse.data.nonastreda import metas_by_tool

    by_tool = metas_by_tool(dataset.metas)
    got = [tool_labels(dataset.timelines[t], by_tool[t], rule).ttf_position for t in dataset.tools]
    assert got == expected
