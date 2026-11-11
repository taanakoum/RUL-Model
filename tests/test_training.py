"""The training loop (PROJECT_PLAN.md part P14) and studies (P20)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from rulfuse.config import FitConfig
from rulfuse.study import Study, aggregate, decide, read_validation_rows, write_manifest
from rulfuse.train import _lr, fit
from tests.toy_engine import TINY, build_toy


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return build_toy(tmp_path_factory.mktemp("toy_train"))


def test_learning_rate_schedule_phases():
    cfg = FitConfig(lr=1e-3, warmup_steps=20, phase_a_steps=100, max_steps=500)
    assert np.isclose(_lr(10, cfg, 100, 500), 5e-4)
    assert np.isclose(_lr(100, cfg, 100, 500), 1e-3)
    assert np.isclose(_lr(101, cfg, 100, 500), 1e-3)
    assert _lr(500, cfg, 100, 500) < 2e-5
    assert np.isclose(_lr(10, cfg, 0, 400), 5e-4, rtol=0.01)  # no phase A: warm-up inside phase B


def test_fit_selects_the_smoothed_minimum_and_writes_its_outputs(toy, tmp_path):
    fold = toy.folds[0]
    sel = fit(
        toy.store, toy.signal, toy.image, fold, 0, TINY, tmp_path / "fit", log=lambda *_: None
    )
    scores = [e["val_mae_s"] for e in sel["evaluations"]]
    smoothed = [np.mean(scores[k - 2 : k + 1]) for k in range(2, len(scores))]
    assert sel["selected_eval"] == int(np.argmin(smoothed)) + 1
    for name in ("model.pt", "predictions.parquet", "diagnostics.json", "selection.json", "DONE"):
        assert (tmp_path / "fit" / name).is_file()
    assert not (tmp_path / "fit" / "checkpoint.pt").exists()
    roles = set(pq.read_table(tmp_path / "fit" / "predictions.parquet").column("role").to_pylist())
    assert roles == {"validation", "test", "train"}
    again = fit(toy.store, toy.signal, toy.image, fold, 0, TINY, tmp_path / "fit")
    assert again == sel  # DONE: skipped


def test_same_seed_same_weights_and_resume_after_a_kill(toy, tmp_path):
    fold = toy.folds[1]
    quiet = lambda *_: None  # noqa: E731
    fit(toy.store, toy.signal, toy.image, fold, 3, TINY, tmp_path / "a", log=quiet)
    fit(toy.store, toy.signal, toy.image, fold, 3, TINY, tmp_path / "b", log=quiet)
    stopped = fit(
        toy.store,
        toy.signal,
        toy.image,
        fold,
        3,
        TINY,
        tmp_path / "c",
        log=quiet,
        stop_after_steps=17,
    )
    assert stopped == {"interrupted_at": 17}
    assert (tmp_path / "c" / "checkpoint.pt").is_file()
    fit(toy.store, toy.signal, toy.image, fold, 3, TINY, tmp_path / "c", log=quiet)
    a, b, c = (torch.load(tmp_path / x / "model.pt", weights_only=True) for x in "abc")
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(torch.equal(a[k], c[k]) for k in a)


def test_validation_reader_never_loads_test_rows(tmp_path):
    path = tmp_path / "p.parquet"
    pq.write_table(
        pa.table(
            {"role": ["validation", "test", "train"], "tau_s": [1.0, 2.0, 3.0], "mu_s": [1, 2, 3]}
        ),
        path,
    )
    rows = read_validation_rows(path)
    assert list(rows["role"]) == ["validation"]


def _rows(maes: dict[str, list]) -> list[dict]:
    return [
        {
            "arm": arm,
            "fold_test": f"T{i}",
            "validation": f"V{i:02d}",
            "done": True,
            "window_mae_s": v,
        }
        for arm, values in maes.items()
        for i, v in enumerate(values)
    ]


def test_gate_decisions(tmp_path):
    spec = {
        "arms": {
            "F1-a": {"fusion": "F1", "lr": 1e-3},
            "F1-b": {"fusion": "F1", "lr": 3e-3},
            "F0-a": {"fusion": "F0", "lr": 1e-3},
        },
        "decision": {"type": "lr_then_family", "threshold_s": 10, "cost": {"F0": 0, "F1": 1}},
    }
    study = Study("g", spec)
    rows = _rows({"F1-a": [100] * 10, "F1-b": [90] * 10, "F0-a": [94] * 10})
    out = decide(study, rows, lambda a: FitConfig().with_overrides(spec["arms"][a]))
    assert out["adopted"] == {"fusion": "F0", "lr": 1e-3}  # within 5 s: the interpretable F0
    spec2 = {
        "arms": {
            "default": {},
            "no-aug": {"trajectory_aug": False},
            "rel": {"token_mode": "relative"},
        },
        "decision": {
            "type": "options",
            "threshold_s": 10,
            "default": "default",
            "components": [{"arm": "no-aug", "field": "trajectory_aug"}],
            "peers": ["default", "rel"],
            "peer_field": "token_mode",
        },
    }
    study2 = Study("g2", spec2)
    rows2 = _rows({"default": [100] * 10, "no-aug": [105] * 10, "rel": [80] * 10})
    out2 = decide(study2, rows2, lambda a: FitConfig().with_overrides(spec2["arms"][a]))
    assert out2["adopted"] == {"trajectory_aug": False, "token_mode": "relative"}
    spec3 = {
        "arms": {"default": {}, "d32": {"d_model": 32}, "d96": {"d_model": 96}},
        "decision": {"type": "one_factor", "threshold_s": 10, "default": "default"},
    }
    rows3 = _rows({"default": [100] * 10, "d32": [85] * 10, "d96": [80] * 7 + [120] * 3})
    out3 = decide(Study("s", spec3), rows3, lambda a: FitConfig())
    assert out3["adopted"] == {"d_model": 32}  # d96 wins only 7 but its mean gain is 8 s
    incomplete = decide(
        study, rows[:-1] + [{"arm": "F0-a", "fold_test": "T9", "done": False}], None
    )
    assert incomplete["complete"] is False


def test_manifest_and_aggregate_on_a_toy_study(toy, tmp_path):
    spec = {
        "store": "toy/x",
        "base": TINY.to_dict(),
        "seeds": [0, 1],
        "arms": {"tiny": {}},
        "folds": [toy.folds[0].test],
        "decision": {"type": "none"},
    }
    study = Study("toystudy", spec)
    path = write_manifest(study, toy.store.unit_names, tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[1].split("\t") == ["0", "tiny", toy.folds[0].test, "0,1"]
    from rulfuse.study import run_task

    run_task(
        study,
        0,
        "cpu",
        tmp_path,
        log=lambda *_: None,
        store=toy.store,
        signal_features=toy.signal,
        image_features=toy.image,
    )
    decision = aggregate(study, tmp_path, store=toy.store)
    assert decision["complete"] and decision["validation_only"]
    summary = (tmp_path / "toystudy" / "summary.csv").read_text(encoding="utf-8")
    assert "window_mae_s" in summary
    assert json.loads((tmp_path / "toystudy" / "decision.json").read_text())["study"] == "toystudy"
