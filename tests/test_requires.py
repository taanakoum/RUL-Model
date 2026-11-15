"""A study with ``requires`` runs only if the inherited decisions adopted those values (D31:
Gate 2b-fix runs only if gate2a-fix chose F0); otherwise it writes a 'skipped' decision that
passes the earlier adoptions on."""

from __future__ import annotations

import json

import pytest

from rulfuse.study import Study, write_manifest

pytestmark = [pytest.mark.integration, pytest.mark.requirements("P20", "D31")]

UNITS = [f"T{i}" for i in range(1, 6)]


def _runs(tmp_path, fusion):
    (tmp_path / "first").mkdir()
    adopted = {"fusion": fusion, "lr": 0.003}
    (tmp_path / "first" / "decision.json").write_text(json.dumps({"adopted": adopted}))
    return Study(
        "second",
        {
            "inherits": ["first"],
            "requires": {"fusion": "F0"},
            "seeds": [0],
            "arms": {"default": {}, "no-phase-a": {"phase_a": False}},
            "decision": {"type": "none"},
        },
    )


@pytest.mark.parametrize("fusion", ["F0", "F1"])
def test_requires_runs_or_skips(tmp_path, fusion):
    study = _runs(tmp_path, fusion)
    path = write_manifest(study, UNITS, tmp_path)
    if fusion == "F0":
        assert path is not None and len(path.read_text().splitlines()) == 1 + 2 * len(UNITS)
        assert study.skip_reason(tmp_path) is None
        return
    assert path is None
    decision = json.loads((tmp_path / "second" / "decision.json").read_text())
    assert decision["complete"] and "F1" in decision["skipped"]
    assert decision["adopted"] == {"fusion": "F1", "lr": 0.003}  # passed on unchanged
    assert not (tmp_path / "second" / "manifest.tsv").exists()
    # a later study inheriting it gets the earlier adoptions
    later = Study(
        "third", {"inherits": ["second"], "arms": {"x": {}}, "decision": {"type": "none"}}
    )
    assert later.config("x", tmp_path).fusion == "F1"
