"""Results of the B6 study: progress, scores, pre-registered choices and a report
(docs/OMNIFUSER_RUL_PLAN.md §7-§8, §10 stage E; PROJECT_PLAN.md D15).

``write_results`` reads the arm folders a study has written so far (finished or not) and
writes ``<runs root>/<dataset>-<store>/results/``:

* ``progress.json``: per arm, the fits done out of folds × seeds, stale fits (other
  settings), the missing (fold, seed) pairs, the selected epochs and the fit times;
* ``curves.csv``: the validation MAE of every epoch of every fit (training diagnostics);
* ``validation_per_unit.csv``, ``analysis-validation.json``: the seed-ensemble scores of
  every arm on each fold's validation unit, with B2 on the same points;
* ``choices.json``: the comparisons of the modality study fixed on the validation units
  (best single arm, best pair, best photos-only pair, best force + photo pair, and v(∅)),
  written before any test read so the test cannot choose its own comparison;
* with ``after_freeze`` (on or after the freeze date), ``test_per_unit.csv`` and
  ``analysis-test.json``: test scores, Shapley values and the paired tests, plus a check
  that the choices used equal the ones recorded before;
* ``RESULTS.md``: all of it as tables.

Test scores stay sealed before the freeze (``aggregate.check_seal``), and a test read
needs every arm complete.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from rulfuse.core.store import SegmentStore
from rulfuse.omnifuser_rul import aggregate
from rulfuse.omnifuser_rul.study import analyse, modality_arms
from rulfuse.omnifuser_rul.train import FORCE, ArmSpec, arm_dir


class IncompleteStudy(RuntimeError):
    """A test read was requested while some arm still misses fits."""


def arm_progress(store: SegmentStore, root: Path, spec: ArmSpec) -> dict:
    folder = arm_dir(root, store, spec)
    n_folds = len(store.unit_names)
    expected = [(f, s) for f in range(1, n_folds + 1) for s in spec.seeds]
    done: dict[tuple[int, int], dict] = {}
    stale = 0
    for path in sorted(folder.glob("fold-*/seed-*/fit.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("spec") != spec.signature():
            stale += 1
            continue
        done[(int(record["fold"]), int(record["seed"]))] = record
    epochs = [r["selected_epoch"] for r in done.values()]
    seconds = [r["seconds"] for r in done.values()]
    return {
        "arm": spec.name,
        "folder": str(folder),
        "expected": len(expected),
        "done": sum(key in done for key in expected),
        "stale": stale,
        "missing": [f"fold-{f:02d}/seed-{s}" for f, s in expected if (f, s) not in done],
        "complete": all(key in done for key in expected),
        "selected_epoch": {
            "median": float(np.median(epochs)) if epochs else None,
            "min": int(min(epochs)) if epochs else None,
            "max": int(max(epochs)) if epochs else None,
            # the centre of the last triple: the curve was still falling when training ended
            "best_triple_at_end": int(sum(e == spec.epochs - 1 for e in epochs)),
        },
        "fit_seconds": {
            "median": float(np.median(seconds)) if seconds else None,
            "total": float(np.sum(seconds)) if seconds else 0.0,
        },
        "_fits": done,
    }


def _curve_rows(progress: Sequence[dict]) -> list[dict]:
    rows = []
    for arm in progress:
        for (fold, seed), record in sorted(arm["_fits"].items()):
            for epoch, value in enumerate(record["validation_mae_s"], start=1):
                rows.append(
                    {
                        "arm": arm["arm"],
                        "fold": fold,
                        "seed": seed,
                        "epoch": epoch,
                        "validation_weighted_mae_s": round(float(value), 3),
                        "selected": epoch == record["selected_epoch"],
                    }
                )
    return rows


def _per_unit_rows(scores: dict, b2: dict[str, float] | None) -> list[dict]:
    rows = []
    for arm, result in scores.items():
        for unit, s in result["per_unit"].items():
            row = {"arm": arm, "unit": unit, **{k: round(float(v), 3) for k, v in s.items()}}
            if b2 is not None:
                row["b2_window_mae"] = round(float(b2[unit]), 3)
            rows.append(row)
    return rows


def preregistered_choices(
    validation: dict, b2_validation: dict[str, float], players: Sequence[str], base: ArmSpec
) -> dict:
    """The study's comparisons, chosen on the validation units only (the same rule as
    ``aggregate.modality_report``: the lowest mean validation window MAE)."""

    def mean(spec: ArmSpec) -> float | None:
        result = validation.get(spec.name)
        return None if result is None else result["summary"]["window_mae"]["mean"]

    def best(coalitions: list[tuple[str, ...]]) -> list[str] | None:
        scored = [(mean(replace(base, players=c)), c) for c in coalitions]
        scored = [(m, c) for m, c in scored if m is not None]
        return sorted(min(scored)[1]) if scored else None

    players = list(players)
    photos = [p for p in players if p != FORCE]
    choices = {
        "rule": "lowest mean validation window MAE of the seed ensemble",
        "best_single": best([(p,) for p in players]),
        "best_pair": best(list(combinations(players, 2))),
    }
    if FORCE in players and len(photos) >= 2:
        choices["best_photos_pair"] = best(list(combinations(photos, 2)))
        choices["best_force_photo_pair"] = best([(FORCE, p) for p in photos])
    usage_only = mean(replace(base, players=()))
    b2_mean = float(np.mean(list(b2_validation.values())))
    if usage_only is not None:
        choices["empty_coalition"] = (
            "B2" if b2_mean <= usage_only else replace(base, players=()).name
        )
    return choices


def _fmt(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}"


def _score_table(scores: dict, progress: dict[str, dict], b2: dict[str, float]) -> list[str]:
    lines = [
        "| Arm | Fits | Window MAE (s), mean ± sd over units | Pooled | Whole-life MAE (s) | "
        "Better than B2 (units) | Mean gain over B2 (s) | Rising steps | Mean rise (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    b2_mean = float(np.mean(list(b2.values())))
    lines.append(f"| B2 (usage rule) | – | {_fmt(b2_mean)} | – | – | – | – | 0 | 0 |")
    for arm, result in scores.items():
        s = result["summary"]
        per_unit = {u: v["window_mae"] for u, v in result["per_unit"].items() if "window_mae" in v}
        gains = [b2[u] - per_unit[u] for u in per_unit if u in b2]
        p = progress.get(arm, {})
        fits = f"{p.get('done', '?')}/{p.get('expected', '?')}"
        lines.append(
            f"| {arm} | {fits} | {_fmt(s['window_mae']['mean'])} ± {_fmt(s['window_mae']['sd'])} "
            f"| {_fmt(s['window_mae']['pooled'])} | {_fmt(s['whole_life_mae']['mean'])} "
            f"| {sum(g > 0 for g in gains)}/{len(gains)} "
            f"| {_fmt(float(np.mean(gains)) if gains else None)} "
            f"| {_fmt(s.get('increase_rate', {}).get('mean'), 3)} "
            f"| {_fmt(s.get('mean_increase_s', {}).get('mean'), 2)} |"
        )
    return lines


def _markdown(results: dict) -> str:
    p = results["progress"]
    done = sum(a["done"] for a in p)
    expected = sum(a["expected"] for a in p)
    lines = [
        "# B6 OmniFuser-RUL: results",
        "",
        f"Store `{results['store']}`; players {', '.join(results['players']) or '(none)'}; "
        f"written {results['written']}.",
        "",
        f"**Progress:** {sum(a['complete'] for a in p)} of {len(p)} arms complete; "
        f"{done} of {expected} fits.",
        "",
        "## Fits",
        "",
        "| Arm | Fits done | Stale | Selected epoch (median, range) | Best triple at the end "
        "| Median fit (s) | Missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in p:
        e = a["selected_epoch"]
        missing = ", ".join(a["missing"][:4]) + (" …" if len(a["missing"]) > 4 else "")
        lines.append(
            f"| {a['arm']} | {a['done']}/{a['expected']} | {a['stale']} | "
            f"{_fmt(e['median'], 0)} ({e['min'] if e['min'] is not None else '–'}-"
            f"{e['max'] if e['max'] is not None else '–'}) | {e['best_triple_at_end']} | "
            f"{_fmt(a['fit_seconds']['median'], 0)} | {missing or '–'} |"
        )
    lines += [
        "",
        "Many fits whose best triple of epochs is the last one means validation error was",
        "still falling when training stopped; `curves.csv` has every curve.",
        "",
        "## Validation units (any time)",
        "",
        "Each fold's validation unit, scored with the 3-seed ensemble on the evaluation grid's",
        "points (the project's 10 s points); window = RUL ≤ H. Rising steps: the share of",
        "consecutive prediction points (every point, 1 s apart for B6) where the predicted RUL",
        "rises; mean rise: its average positive change. B2 never rises.",
        "",
        *_score_table(results["validation"], {a["arm"]: a for a in p}, results["b2_validation"]),
        "",
        "## Comparisons fixed on validation (before any test read)",
        "",
    ]
    for key, value in results["choices"].items():
        lines.append(f"- **{key.replace('_', ' ')}:** {value}")
    lines.append("")
    if "test" not in results:
        lines += [
            "## Test units",
            "",
            f"Sealed until {aggregate.FREEZE} (D15). The test predictions are written by every "
            "fit and copied with these results; rerun with `--after-freeze` on or after that "
            "date to score them.",
        ]
        return "\n".join(lines) + "\n"
    lines += [
        "## Test units (after the freeze)",
        "",
        *_score_table(results["test"], {a["arm"]: a for a in p}, results["b2_test"]),
        "",
        f"v(∅) is **{results['empty_coalition']}** (the better on validation of B2 and the "
        "usage-only arm).",
        "",
    ]
    modality = results.get("modality", {})
    if modality:
        lines += ["### Shapley values (window MAE, s; positive = the player lowers the error)", ""]
        lines += ["| Player | φ mean | 95 % interval (unit bootstrap) |", "|---|---|---|"]
        for player, phi in modality["shapley"].items():
            lo, hi = phi["ci95"]
            lines.append(f"| {player} | {_fmt(phi['mean'])} | {_fmt(lo)} to {_fmt(hi)} |")
        lines += [
            "",
            "### Paired tests (one-sided Wilcoxon over units; justified: gain ≥ 10 s and p < 0.05)",
            "",
        ]
        lines += [
            "| Comparison | Mean gain (s) | Units better / worse | p | Justified |",
            "|---|---|---|---|---|",
        ]
        names = {
            "pair_vs_single": (
                f"best pair {modality.get('best_pair')} "
                f"vs best single {modality.get('best_single')}"
            ),
            "all_vs_pair": "all players vs the best pair",
            "force_photo_vs_photos": "best force + photo vs best photos-only pair",
        }
        for key, label in names.items():
            if key in modality:
                t = modality[key]
                lines.append(
                    f"| {label} | {_fmt(t['mean_gain_s'])} "
                    f"| {t['units_better']} / {t['units_worse']} "
                    f"| {t['p_one_sided']:.3g} | {'yes' if t['justified'] else 'no'} |"
                )
    lines += [
        "",
        "Choices used equal the ones recorded before the test read: "
        f"**{results.get('choices_match')}**.",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    store: SegmentStore,
    root: Path,
    players: Sequence[str],
    base: ArmSpec | None = None,
    after_freeze: bool = False,
    out: Path | None = None,
    today: date | None = None,
) -> Path:
    base = base or ArmSpec(players=())
    out = out or root / f"{store.dataset}-{store.signature}" / "results"
    out.mkdir(parents=True, exist_ok=True)
    arms = modality_arms(players, base)
    progress = [arm_progress(store, root, spec) for spec in arms]
    _write_csv(_curve_rows(progress), out / "curves.csv")
    clean = [{k: v for k, v in a.items() if k != "_fits"} for a in progress]
    (out / "progress.json").write_text(json.dumps(clean, indent=1), encoding="utf-8")

    report = analyse(store, root, players, base)  # validation only
    choices = preregistered_choices(report["validation"], report["b2_validation"], players, base)
    results: dict = {
        "store": f"{store.dataset}/{store.signature}",
        "players": list(players),
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "progress": clean,
        "validation": report["validation"],
        "b2_validation": report["b2_validation"],
        "choices": choices,
    }
    (out / "analysis-validation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _write_csv(
        _per_unit_rows(report["validation"], report["b2_validation"]),
        out / "validation_per_unit.csv",
    )
    choices_path = out / "choices.json"
    if not after_freeze:
        record = {
            "recorded": results["written"],
            "complete": all(a["complete"] for a in clean),
            **choices,
        }
        choices_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    else:
        aggregate.check_seal("test", after_freeze, today)
        incomplete = [a["arm"] for a in clean if not a["complete"]]
        if incomplete:
            raise IncompleteStudy(
                f"arms missing fits: {incomplete}; finish them before the test read"
            )
        recorded = (
            json.loads(choices_path.read_text(encoding="utf-8")) if choices_path.is_file() else {}
        )
        full = analyse(store, root, players, base, after_freeze=True, today=today)
        b2_test = aggregate.b2_by_unit(store, "test")
        modality = full.get("modality", {})
        used = {"best_single": modality.get("best_single"), "best_pair": modality.get("best_pair")}
        results.update(
            test=full["test"],
            b2_test=b2_test,
            empty_coalition=full.get("empty_coalition"),
            modality=modality,
            choices_recorded_before=recorded or None,
            choices_match=bool(recorded)
            and all(recorded.get(k) == v for k, v in used.items() if v is not None),
        )
        (out / "analysis-test.json").write_text(json.dumps(full, indent=1), encoding="utf-8")
        _write_csv(_per_unit_rows(full["test"], b2_test), out / "test_per_unit.csv")
    (out / "RESULTS.md").write_text(_markdown(results), encoding="utf-8")
    return out
