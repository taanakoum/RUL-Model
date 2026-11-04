"""P16 (part) Usage baselines B0-B2 (PROJECT_PLAN.md §2.7, §8)."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import tool_metas

from rulfuse.baselines import b0, b1, b2, b2_interval, causal_elapsed_passes, score_usage_baselines
from rulfuse.data.folds import leave_one_tool_out_folds
from rulfuse.data.labels import tool_labels
from rulfuse.data.timeline import DEFAULT_PASS_SECONDS, build_tool_timeline


def test_b2_by_hand():
    ttf = np.array([10.0, 20.0, 30.0])
    # At t = 15 the tools with TTF 20 and 30 are alive: residuals 5 and 15.
    assert b2(ttf, np.array([15.0]), cap=15)[0] == pytest.approx(10.0)
    assert b2(ttf, np.array([15.0]), cap=8)[0] == pytest.approx(6.5)  # cap inside: 5, 8
    assert b2(ttf, np.array([35.0]), cap=15)[0] == 0.0  # nobody alive
    assert b1(ttf, np.array([15.0, 25.0]), cap=15).tolist() == [5.0, 0.0]
    assert b0(3, 15).tolist() == [7.5, 7.5, 7.5]
    lower, upper = b2_interval(ttf, np.array([15.0]), cap=15, level=0.9)
    assert lower[0] == pytest.approx(5.5) and upper[0] == pytest.approx(14.5)


def test_elapsed_passes_use_only_past_durations():
    metas = tool_metas(["sharp"] * 7 + ["dulled"] * 3)
    durations = {m.record_id: 80.0 + 10 * i for i, m in enumerate(metas)}  # 80, 90, 100, ...
    labels = tool_labels(build_tool_timeline(metas, durations), metas)
    elapsed = causal_elapsed_passes(labels)
    p = labels.points
    first = p.position == 1
    np.testing.assert_allclose(
        elapsed[first], np.minimum(p.elapsed_in_pass[first] / DEFAULT_PASS_SECONDS, 1)
    )
    second = p.position == 2  # mean of the first pass only (80 s), not its own 90 s
    np.testing.assert_allclose(elapsed[second], 1 + np.minimum(p.elapsed_in_pass[second] / 80.0, 1))


@pytest.mark.data
def test_real_baselines_reproduce_the_plan(dataset):
    summary = score_usage_baselines(dataset.labels, list(leave_one_tool_out_folds(dataset.tools)))[
        "summary"
    ]
    assert summary["B0"]["pass"]["mae"] == pytest.approx(4.13, abs=0.01)
    assert summary["B1"]["pass"]["mae"] == pytest.approx(5.17, abs=0.01)
    assert summary["B2"]["pass"]["mae"] == pytest.approx(3.82, abs=0.01)
    assert summary["B0"]["sec"]["mae"] == pytest.approx(412.0, abs=0.5)
    assert summary["B1"]["sec"]["mae"] == pytest.approx(512.5, abs=0.5)
    assert summary["B2"]["sec"]["mae"] == pytest.approx(467.8, abs=0.5)
