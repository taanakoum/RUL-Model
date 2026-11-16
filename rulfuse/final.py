"""Stage 7: the test-tool scores of the frozen design (PROJECT_PLAN.md §4.6, D15).

This is the only module that reads prediction rows with ``role == "test"``. Every entry point
calls :func:`check_seal` first, which refuses unless

* ``configs/final.json`` exists and its hash is intact, and
* (for a study) the study's design equals the frozen one.

Nothing is trained here: the fits are the ones Gate 3 made (one seed: the same computation as
the final fits, J10), scored on the tool each fold left out.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from rulfuse import paths
from rulfuse.core.store import stable_hash
from rulfuse.gate3 import fold_calibration, stack_seeds
from rulfuse.study import Study, fit_dir, open_store, study_folds


class SealedError(RuntimeError):
    """The test tools may not be read yet."""


def load_frozen(path: Path | None = None) -> dict:
    path = path or paths.REPO_ROOT / "configs" / "final.json"
    if not path.is_file():
        raise SealedError(f"{path} does not exist: run `rulfuse freeze` first (D15)")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in frozen.items() if k not in ("hash", "freeze_date")}
    if stable_hash(body) != frozen.get("hash"):
        raise SealedError(f"{path} was edited after the freeze (hash mismatch)")
    return frozen


def check_seal(frozen_path: Path | None = None, today: date | None = None) -> dict:
    """Return the frozen design, or refuse (D15)."""
    # The freeze date no longer gates reading (user decision, 2026-11-16): once the design is
    # frozen in configs/final.json the test tools can be scored. ``today`` is kept for callers.
    del today
    return load_frozen(frozen_path)


def check_design(study: Study, arm: str, frozen: dict, runs: Path | None) -> None:
    if study.config(arm, runs).to_dict() != frozen["design"]:
        raise SealedError(f"arm {arm!r} of {study.name} is not the frozen design")


def read_test_rows(path: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("role", "=", "test")])
    return {c: table.column(c).to_numpy(zero_copy_only=False) for c in table.column_names}


def read_test_tool(study: Study, arm: str, fold, runs: Path | None) -> dict | None:
    """The fold's test tool, seeds stacked; None if a fit is unfinished."""
    parts = []
    for seed in study.arm_seeds(arm):
        folder = fit_dir(study.run_dir(runs), arm, fold, seed)
        if not (folder / "DONE").is_file():
            return None
        parts.append(read_test_rows(folder / "predictions.parquet"))
    tool = stack_seeds(parts)
    if not set(np.unique(parts[0]["unit"])) <= {fold.test}:
        raise ValueError(f"{fold.test}: test rows belong to {np.unique(parts[0]['unit'])}")
    return tool


def window_mae(tool: dict) -> float:
    from rulfuse.uncertainty import mixture_moments

    mean, _ = mixture_moments(tool["mu"], tool["sigma"], tool["nu"])
    w = tool["window"]
    return float(np.mean(np.abs(mean[w] - tool["rul"][w])))


def final_scores(
    study: Study,
    arm: str,
    runs: Path | None = None,
    frozen_path: Path | None = None,
    today: date | None = None,
) -> dict:
    """Headline numbers per test tool for the frozen design: unfiltered and filtered window
    MAE, B0, B2, and the frozen calibration's coverage and Winkler score. Needs the study to
    have inner fits (Gate 3)."""
    from rulfuse.baselines import b2

    frozen = check_seal(frozen_path, today)
    check_design(study, arm, frozen, runs)
    store = open_store(study.spec)
    horizon = store.settings.horizon_s
    choice = frozen["calibration"]
    level = float(choice["level"])
    rows = []
    for fold in study_folds(study, store.unit_names):
        tool = read_test_tool(study, arm, fold, runs)
        if tool is None:
            rows.append({"test": fold.test, "done": False})
            continue
        cal = fold_calibration(study, arm, fold, store, runs, level, scored=tool)
        if not cal.get("done"):
            rows.append({"test": fold.test, "done": False})
            continue
        w = tool["window"]
        failures = np.array(
            [store.units[u]["failure_s"] for u in fold.train if store.units[u]["failure_s"]]
        )
        method = cal["methods"][choice["method"]]
        flt = cal["filter"]
        rows.append(
            {
                "test": fold.test,
                "done": True,
                "points": int(w.sum()),
                "window_mae_s": cal["window_mae_s"],
                "filtered_window_mae_s": flt["window_mae_s"],
                "b0_window_mae_s": float(np.mean(np.abs(horizon / 2 - tool["rul"][w]))),
                "b2_window_mae_s": float(
                    np.mean(np.abs(b2(failures, tool["tau"][w], horizon) - tool["rul"][w]))
                ),
                f"coverage_{choice['method']}": method["coverage"],
                f"winkler_{choice['method']}": method["winkler"],
                "coverage_filter": flt["coverage"],
                "winkler_filter": flt["winkler"],
            }
        )
    return {"frozen_hash": frozen["hash"], "calibration": choice, "rows": rows}


def write_final(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "test_per_unit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    done = [r for r in rows if r.get("done")]
    lines = [
        "# Final result: test tools (frozen design)",
        "",
        f"Frozen design hash `{result['frozen_hash']}`; calibration {result['calibration']}.",
        "",
    ]
    if len(done) < len(rows):
        lines += [f"**Incomplete: {len(rows) - len(done)} folds missing.**", ""]
    if done:

        def mean(k):
            return float(np.mean([r[k] for r in done]))

        lines += [
            "| Quantity | Mean over test tools |",
            "|---|---|",
            f"| Window MAE, model (s) | {mean('window_mae_s'):.1f} |",
            f"| Window MAE, with causal filter (s) | {mean('filtered_window_mae_s'):.1f} |",
            f"| B0, constant (s) | {mean('b0_window_mae_s'):.1f} |",
            f"| B2, usage only (s) | {mean('b2_window_mae_s'):.1f} |",
            f"| Filter coverage / Winkler (s) | {mean('coverage_filter'):.3f} / "
            f"{mean('winkler_filter'):.0f} |",
            "",
            "Per tool: `test_per_unit.csv`.",
        ]
    path = out_dir / "FINAL.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def shapley_test(
    study: Study,
    runs: Path | None = None,
    frozen_path: Path | None = None,
    today: date | None = None,
) -> dict:
    """Shapley values on the test tools of a Shapley study (§11.1), after the seal check."""
    from rulfuse.shapley import EMPTY, coalition_key, per_tool_values, render, summarise

    frozen = check_seal(frozen_path, today)
    store = open_store(study.spec)
    by_coalition: dict = {}
    full_arm = None
    for arm in study.arms:
        cfg = study.config(arm, runs)
        key = coalition_key("F" if p == "signal" else p[0].upper() for p in cfg.players)
        if "signal" in cfg.players and len(cfg.players) == len(frozen["design"]["players"]):
            check_design(study, arm, frozen, runs)
            full_arm = arm
        by_coalition[key] = arm
    if full_arm is None:
        raise SealedError("no arm of the study is the frozen design")
    from rulfuse.baselines import b2

    folds = sorted(study_folds(study, store.unit_names), key=lambda f: f.test)
    horizon = store.settings.horizon_s
    mae = {c: [] for c in by_coalition}
    empty = []
    tools = []
    for fold in folds:
        tools.append(fold.test)
        failures = np.array(
            [store.units[u]["failure_s"] for u in fold.train if store.units[u]["failure_s"]]
        )
        w = None
        for c, arm in by_coalition.items():
            tool = read_test_tool(study, arm, fold, runs)
            if tool is None:
                return {"complete": False}
            mae[c].append(window_mae(tool))
            if arm == full_arm:
                w = tool
        assert w is not None
        empty.append(
            float(
                np.mean(
                    np.abs(b2(failures, w["tau"][w["window"]], horizon) - w["rul"][w["window"]])
                )
            )
        )
    arrays = {c: np.array(v) for c, v in mae.items()}
    arrays[EMPTY] = np.array(empty)
    players = list(max(by_coalition, key=len))
    summary = summarise(per_tool_values(arrays, players), tools)
    summary.update({"complete": True, "scope": "test", "study": study.name})
    out = study.run_dir(runs)
    (out / "shapley_test.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "SHAPLEY_TEST.md").write_text(render(summary, "test"), encoding="utf-8")
    return summary
