"""P6 Folds (PROJECT_PLAN.md D1, §4.1)."""

from __future__ import annotations

import pytest

from rulfuse.core.folds import leave_one_unit_out

TOOLS = [f"T{i}" for i in range(1, 11)]


def test_ten_disjoint_cyclic_folds():
    folds = leave_one_unit_out(TOOLS)
    assert len(folds) == 10
    assert [f.test for f in folds] == TOOLS  # the adapter's order is kept
    assert [f.validation for f in folds] == [*TOOLS[1:], "T1"]
    for fold in folds:
        roles = [set(fold.train), {fold.validation}, {fold.test}]
        assert sum(len(r) for r in roles) == 10 and set().union(*roles) == set(TOOLS)


def test_order_is_the_adapters_and_duplicates_collapse():
    folds = leave_one_unit_out(["b", "a", "c", "a"])
    assert [f.test for f in folds] == ["b", "a", "c"]
    assert [f.validation for f in folds] == ["a", "c", "b"]


def test_needs_three_tools():
    with pytest.raises(ValueError):
        leave_one_unit_out(["T1", "T2"])
