"""P6 Folds (PROJECT_PLAN.md D1, §4.1)."""

from __future__ import annotations

import pytest

from rulfuse.data.folds import leave_one_tool_out_folds

TOOLS = [f"T{i}" for i in range(1, 11)]


def test_ten_disjoint_cyclic_folds():
    folds = list(leave_one_tool_out_folds(reversed(TOOLS)))
    assert len(folds) == 10
    assert [f.test for f in folds] == TOOLS
    assert [f.validation for f in folds] == [*TOOLS[1:], "T1"]
    for fold in folds:
        roles = [set(fold.train_tools), {fold.validation}, {fold.test}]
        assert sum(len(r) for r in roles) == 10 and set().union(*roles) == set(TOOLS)


def test_needs_three_tools():
    with pytest.raises(ValueError):
        list(leave_one_tool_out_folds(["T1", "T2"]))
