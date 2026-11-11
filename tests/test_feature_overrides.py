"""D24: photo types readmitted after the Stage-3 feature check (configs/, overrides file)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rulfuse.features.check import admitted_image_types

OVERRIDES = Path(__file__).resolve().parents[1] / "configs" / "feature_check_overrides.json"
# The decisions of the Ruche report nonastreda-3e357b5a23ff.json (2026-11-11)
RUCHE = {
    "dataset": "nonastreda",
    "decisions": {
        "signals_go": True,
        "image_types_kept": [],
        "image_types_excluded": {
            "chip": "leaks the stage label",
            "tool": "leaks the stage label",
            "work": "leaks the stage label",
        },
    },
}


def test_the_recorded_decision_readmits_tool_only():
    overrides = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    assert admitted_image_types(RUCHE, overrides) == ["tool"]
    entry = overrides["nonastreda"]["admit"]["tool"]
    assert {"decided", "report", "reason"} <= set(entry)


def test_without_overrides_the_check_decides_alone():
    assert admitted_image_types(RUCHE) == []
    kept = {**RUCHE, "decisions": {**RUCHE["decisions"], "image_types_kept": ["chip"]}}
    assert admitted_image_types(kept, {"other_dataset": {"admit": {"tool": {}}}}) == ["chip"]


def test_an_override_cannot_invent_a_photo_type():
    with pytest.raises(ValueError, match="did not examine"):
        admitted_image_types(RUCHE, {"nonastreda": {"admit": {"thermal": {}}}})
