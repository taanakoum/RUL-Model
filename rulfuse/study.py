"""Studies: arms, manifests, array tasks, aggregation and gate decisions (PROJECT_PLAN.md
§4.4, §4.6, §7.5, §7.11; parts P14, P20).

A study is ``configs/studies/<name>.json``:

* ``store``: the segment store, as ``<dataset>/<signature>``;
* ``base``: overrides shared by every arm;
* ``inherits``: earlier studies whose ``decision.json`` → ``adopted`` overrides apply first;
* ``arms``: name → overrides;
* ``seeds``;
* ``decision``: the rule that decides the study.

An arm's config is the defaults, then the inherited adoptions, then ``base``, then the arm.

**Workflow.**

1. ``rulfuse manifest --study S`` writes ``runs/S/manifest.tsv`` (one task per arm × fold;
   the seeds run inside a task).
2. ``slurm/array.sbatch`` runs ``rulfuse run-task`` per line; finished fits are skipped.
3. ``rulfuse aggregate --study S`` builds the seed ensembles on the **validation** units, scores
   the window MAE per fold and applies the study's rule. It writes ``summary.csv``,
   ``decision.json`` and ``REPORT.md``.

**Test seal (D15).** The aggregation reads prediction rows with role ``validation`` only;
test rows are filtered out when the file is read, so no test value is ever loaded. Test
scoring belongs to the final study after the freeze (Stage 7), not here.

**Decision types**, each fixed before the study runs (§4.4, §17.3):

* ``lr_then_family`` (Gate 2a): Rule B among the learning rates of each family, then Rule B
  between the families' best, with the cost order of the study (F0 first: its weights
  read directly as modality importance, which is the plan's "more interpretable");
* ``options`` (Gate 2b): for each component the default switches on, Rule A of the default
  against the arm without it. The component stays only if it earns its place. Token modes
  are peers under Rule B. Controls are reported only;
* ``one_factor`` (sweep): each variant against the default under Rule A. Adopted variants
  are combined; if two values of one factor pass, the better one is taken.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rulfuse import paths
from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold, leave_one_unit_out
from rulfuse.core.store import SegmentStore
from rulfuse.stats import rule_a, rule_b

STUDIES_DIR = paths.REPO_ROOT / "configs" / "studies"
FREEZE_DATE = "2026-12-01"


@dataclass
class Study:
    name: str
    spec: dict

    @classmethod
    def load(cls, name: str, root: Path | None = None) -> Study:
        path = (root or STUDIES_DIR) / f"{name}.json"
        return cls(name, json.loads(path.read_text(encoding="utf-8")))

    @property
    def seeds(self) -> list[int]:
        return list(self.spec.get("seeds", [0, 1, 2]))

    @property
    def arms(self) -> dict[str, dict]:
        return dict(self.spec["arms"])

    def arm_seeds(self, arm: str) -> list[int]:
        """The arm's seeds: ``arm_seeds`` in the spec (B2, reported controls on one seed),
        otherwise the study's seeds."""
        return list(self.spec.get("arm_seeds", {}).get(arm, self.seeds))

    def run_dir(self, runs: Path | None = None) -> Path:
        return (runs or paths.runs_dir()) / self.name

    def inherited(self, runs: Path | None = None) -> dict:
        adopted: dict = {}
        for name in self.spec.get("inherits", []):
            path = (runs or paths.runs_dir()) / name / "decision.json"
            if not path.is_file():
                raise FileNotFoundError(f"{path}: aggregate study {name!r} first")
            adopted.update(json.loads(path.read_text(encoding="utf-8"))["adopted"])
        return adopted

    def config(self, arm: str, runs: Path | None = None) -> FitConfig:
        base = FitConfig().with_overrides(self.inherited(runs))
        return base.with_overrides(self.spec.get("base", {})).with_overrides(self.arms[arm])


def study_folds(study: Study, units: list[str]) -> list[Fold]:
    """The study's folds: all of them, or the ``folds`` it names (by test unit)."""
    folds = leave_one_unit_out(units)
    named = study.spec.get("folds")
    return [f for f in folds if f.test in named] if named else folds


def open_store(spec: dict) -> SegmentStore:
    """The study's store: ``store_root`` if the spec gives one, else
    interim/segments/<dataset>/<signature>."""
    if spec.get("store_root"):
        return SegmentStore.open(Path(spec["store_root"]))
    dataset, signature = spec["store"].split("/")
    return SegmentStore.open(paths.interim_dir() / "segments" / dataset / signature)


def fit_dir(study_dir: Path, arm: str, fold: Fold, seed: int) -> Path:
    return study_dir / arm / f"fold-{fold.test}" / f"seed-{seed}"


# ----------------------------------------------------------------------------- manifest


def write_manifest(study: Study, units: list[str], runs: Path | None = None) -> Path:
    for arm in study.arms:
        study.config(arm, runs)  # fails early on an unknown field or a missing inheritance
    out = study.run_dir(runs)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.tsv"
    folds = study_folds(study, units)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["task", "arm", "fold_test", "seeds"])
        task = 0
        for arm in study.arms:
            for fold in folds:
                w.writerow([task, arm, fold.test, ",".join(map(str, study.arm_seeds(arm)))])
                task += 1
    (out / "study.json").write_text(json.dumps(study.spec, indent=1), encoding="utf-8")
    return path


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def find_identical_fit(
    runs: Path, store_name: str, config_key: str, fold: Fold, seed: int, target: Path
) -> Path | None:
    """A finished fit of any study with the same store, configuration, fold and seed (B1):
    fits are deterministic, so it is the fit this task would make."""
    for selection in sorted(runs.glob(f"*/*/fold-{fold.test}/seed-{seed}/selection.json")):
        folder = selection.parent
        if folder.resolve() == target.resolve() or not (folder / "DONE").is_file():
            continue
        if (folder / "reused_from.txt").is_file():
            continue  # copy from originals only
        spec_path = folder.parents[2] / "study.json"
        if not spec_path.is_file():
            continue
        if json.loads(spec_path.read_text(encoding="utf-8")).get("store") != store_name:
            continue
        info = json.loads(selection.read_text(encoding="utf-8"))
        same_roles = info.get("fold_validation") == fold.validation
        if info.get("config_key") == config_key and same_roles and info.get("seed") == seed:
            return folder
    return None


def reuse_fit(source: Path, target: Path) -> dict:
    import shutil

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".copying")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(source, tmp)
    (tmp / "reused_from.txt").write_text(str(source), encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    tmp.rename(target)
    return json.loads((target / "selection.json").read_text(encoding="utf-8"))


def _seed_worker(name: str, spec: dict, task: int, seed: int, device: str, runs, threads: int):
    import torch

    torch.set_num_threads(threads)
    run_task(Study(name, spec), task, device, runs, seeds=[seed])


def run_task(
    study: Study,
    task: int,
    device: str = "cpu",
    runs: Path | None = None,
    log=print,
    store: SegmentStore | None = None,
    signal_features=None,
    image_features=None,
    seeds: list[int] | None = None,
    parallel: bool = False,
    threads: int = 1,
) -> list[dict]:
    """Run every seed of one manifest line.

    * Finished fits are skipped; an interrupted one resumes.
    * A fit identical to a finished one elsewhere (same store, configuration, fold and
      seed) is copied, not rerun (B1).
    * With ``parallel``, the pending seeds run side by side in separate processes on the
      same device (A1). The tokens are prepared once first, so the processes read the
      cached readouts. Results are the same as one after another: every fit seeds itself.
    """
    from rulfuse.features.cache import ImageFeatures, SignalFeatures, find_features
    from rulfuse.tokens import build_fold_data
    from rulfuse.train import fit

    rows = read_manifest(study.run_dir(runs) / "manifest.tsv")
    row = rows[task]
    store = store or open_store(study.spec)
    folds = {f.test: f for f in leave_one_unit_out(store.unit_names)}
    fold = folds[row["fold_test"]]
    config = study.config(row["arm"], runs)
    features_root = Path(study.spec["features_root"]) if study.spec.get("features_root") else None
    if signal_features is None and config.uses_signal:
        signal_features = SignalFeatures.open(
            find_features(store, "signal", study.spec["signal_encoder"], features_root)
        )
    if image_features is None and config.image_kinds:
        image_features = ImageFeatures.open(
            find_features(store, "image", study.spec["image_encoder"], features_root)
        )
    from rulfuse.train import fit_key

    runs_root = runs or paths.runs_dir()
    wanted = [int(s) for s in row["seeds"].split(",")]
    if seeds is not None:
        wanted = [s for s in wanted if s in seeds]
    pending = []
    key = fit_key(config)
    for seed in wanted:
        target = fit_dir(study.run_dir(runs), row["arm"], fold, seed)
        if (target / "DONE").is_file():
            continue
        source = find_identical_fit(runs_root, study.spec["store"], key, fold, seed, target)
        if source is not None:
            log(f"[task {task}] seed {seed}: identical fit found, copied from {source}")
            reuse_fit(source, target)
            continue
        pending.append(seed)
    data = build_fold_data(store, signal_features, image_features, fold, config)
    if parallel and len(pending) > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(
                target=_seed_worker,
                args=(study.name, study.spec, task, s, device, runs, threads),
            )
            for s in pending
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [s for s, p in zip(pending, procs, strict=True) if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"task {task}: seeds {failed} failed (see the log above)")
        pending = []
    out = []
    for seed in wanted:
        target = fit_dir(study.run_dir(runs), row["arm"], fold, seed)
        if seed not in pending:
            out.append(json.loads((target / "selection.json").read_text(encoding="utf-8")))
            continue
        out.append(
            fit(
                store,
                signal_features,
                image_features,
                fold,
                seed,
                config,
                target,
                device,
                log,
                data=data,
            )
        )
    return out


# ----------------------------------------------------------------------------- aggregate


def read_validation_rows(path: Path) -> dict[str, np.ndarray]:
    """The validation rows of a predictions file; test rows are never loaded (D15)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("role", "=", "validation")])
    return {c: table.column(c).to_numpy(zero_copy_only=False) for c in table.column_names}


def ensemble_validation(study_dir: Path, arm: str, fold: Fold, seeds: list[int]) -> dict | None:
    """Seed ensemble of the validation unit: mixture mean (and moments), per point."""
    from rulfuse.uncertainty import mixture_moments

    parts = []
    for seed in seeds:
        path = fit_dir(study_dir, arm, fold, seed) / "predictions.parquet"
        if not (path.parent / "DONE").is_file():
            return None
        parts.append(read_validation_rows(path))
    order = [np.lexsort((p["tau_s"],)) for p in parts]
    parts = [{k: v[o] for k, v in p.items()} for p, o in zip(parts, order, strict=True)]
    mu = np.stack([p["mu_s"] for p in parts])
    sigma = np.stack([p["sigma_s"] for p in parts])
    nu = np.stack([p["nu"] for p in parts])
    mean, var = mixture_moments(mu, sigma, nu)
    first = parts[0]
    return {
        "tau_s": first["tau_s"],
        "rul_s": first["rul_s"],
        "window": first["window"].astype(bool),
        "mu_s": mean,
        "sd_s": np.sqrt(var),
        "seed_window_mae": [
            float(
                np.mean(
                    np.abs(
                        m[first["window"].astype(bool)]
                        - first["rul_s"][first["window"].astype(bool)]
                    )
                )
            )
            for m in mu
        ],
    }


def score_arm(study: Study, arm: str, store: SegmentStore, runs: Path | None = None) -> list[dict]:
    from rulfuse.baselines import b2

    horizon = store.settings.horizon_s
    rows = []
    for fold in study_folds(study, store.unit_names):
        ens = ensemble_validation(study.run_dir(runs), arm, fold, study.arm_seeds(arm))
        if ens is None:
            rows.append(
                {"arm": arm, "fold_test": fold.test, "validation": fold.validation, "done": False}
            )
            continue
        keep = ens["window"]
        failures = np.array(
            [store.units[u]["failure_s"] for u in fold.train if store.units[u]["failure_s"]]
        )
        rul = ens["rul_s"][keep]
        warnings = []
        for seed in study.arm_seeds(arm):
            diag = fit_dir(study.run_dir(runs), arm, fold, seed) / "diagnostics.json"
            if diag.is_file():
                warnings += json.loads(diag.read_text(encoding="utf-8"))["warnings"]
        rows.append(
            {
                "arm": arm,
                "fold_test": fold.test,
                "validation": fold.validation,
                "done": True,
                "points": int(keep.sum()),
                "window_mae_s": float(np.mean(np.abs(ens["mu_s"][keep] - rul))),
                "b2_window_mae_s": float(
                    np.mean(np.abs(b2(failures, ens["tau_s"][keep], horizon) - rul))
                ),
                "b0_window_mae_s": float(np.mean(np.abs(horizon / 2 - rul))),
                "seed_sd_s": float(np.std(ens["seed_window_mae"])),
                "warnings": len(warnings),
            }
        )
    return rows


def _per_unit(rows: list[dict], arm: str) -> np.ndarray:
    mine = sorted(
        (r for r in rows if r["arm"] == arm and r.get("done")), key=lambda r: r["validation"]
    )
    return np.array([r["window_mae_s"] for r in mine])


def decide(study: Study, rows: list[dict], config_of) -> dict:
    rule = study.spec["decision"]
    kind, threshold = rule["type"], float(rule.get("threshold_s", 10.0))
    arms = list(study.arms)
    if not all(r.get("done") for r in rows):
        missing = sorted({f"{r['arm']}/{r['fold_test']}" for r in rows if not r.get("done")})
        return {"complete": False, "missing": missing, "adopted": {}}
    maes = {a: _per_unit(rows, a) for a in arms}
    out: dict = {
        "complete": True,
        "type": kind,
        "threshold_s": threshold,
        "means": {a: float(v.mean()) for a, v in maes.items()},
    }
    adopted: dict = {}
    if kind == "lr_then_family":
        families: dict[str, list[str]] = {}
        for a in arms:
            families.setdefault(config_of(a).fusion, []).append(a)
        best = {}
        for fam, members in families.items():
            res = rule_b({a: maes[a] for a in members}, threshold)
            best[fam] = res["chosen"]
            out[f"rule_b_{fam}"] = res
        cost = {best[f]: float(rule.get("cost", {}).get(f, 0.0)) for f in best}
        res = rule_b({best[f]: maes[best[f]] for f in best}, threshold, cost)
        out["rule_b_family"] = res
        chosen = config_of(res["chosen"])
        adopted = {"fusion": chosen.fusion, "lr": chosen.lr}
    elif kind == "options":
        default = rule["default"]
        for option in rule.get("components", []):
            res = rule_a(maes[option["arm"]], maes[default], threshold)
            out[f"rule_a_{option['field']}"] = res
            if not res["adopted"]:
                adopted[option["field"]] = config_of(option["arm"]).to_dict()[option["field"]]
        peers = rule.get("peers")
        if peers:
            res = rule_b({a: maes[a] for a in peers}, threshold)
            out["rule_b_peers"] = res
            adopted[rule["peer_field"]] = config_of(res["chosen"]).to_dict()[rule["peer_field"]]
        out["controls"] = {a: float(maes[a].mean()) for a in rule.get("controls", [])}
    elif kind == "none":
        pass
    elif kind == "one_factor":
        default = rule["default"]
        passed: dict[str, tuple[float, object]] = {}
        for arm in arms:
            if arm == default:
                continue
            res = rule_a(maes[default], maes[arm], threshold)
            out[f"rule_a_{arm}"] = res
            if res["adopted"]:
                field = next(iter(study.arms[arm]))
                value = study.arms[arm][field]
                if field not in passed or res["gain_mean"] > passed[field][0]:
                    passed[field] = (res["gain_mean"], value)
        adopted = {f: v for f, (_, v) in passed.items()}
    else:
        raise ValueError(f"unknown decision type {kind!r}")
    out["new_adoptions"] = adopted
    out["adopted"] = dict(adopted)
    return out


def aggregate(study: Study, runs: Path | None = None, store: SegmentStore | None = None) -> dict:
    store = store or open_store(study.spec)
    rows = [r for arm in study.arms for r in score_arm(study, arm, store, runs)]
    out_dir = study.run_dir(runs)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    decision = decide(study, rows, lambda a: study.config(a, runs))
    # what later studies inherit: earlier adoptions plus this study's
    decision["adopted"] = {**study.inherited(runs), **decision.get("new_adoptions", {})}
    decision["study"] = study.name
    decision["validation_only"] = True
    (out_dir / "decision.json").write_text(
        json.dumps(decision, indent=1, default=float), encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(render_report(study, rows, decision), encoding="utf-8")
    return decision


def render_report(study: Study, rows: list[dict], decision: dict) -> str:
    lines = [
        f"# Study {study.name}",
        "",
        f"Validation units only (D15); {len(study.seeds)}-seed ensembles; window MAE (s).",
        "",
    ]
    lines += [
        "| Arm | Mean window MAE | Mean B2 | Mean B0 | Folds better than B2 | Seed sd | Warnings |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in study.arms:
        mine = [r for r in rows if r["arm"] == arm and r.get("done")]
        if not mine:
            lines.append(f"| {arm} | not done | | | | | |")
            continue
        m = np.mean([r["window_mae_s"] for r in mine])
        b2m = np.mean([r["b2_window_mae_s"] for r in mine])
        b0m = np.mean([r["b0_window_mae_s"] for r in mine])
        better = sum(r["window_mae_s"] < r["b2_window_mae_s"] for r in mine)
        sd = np.mean([r["seed_sd_s"] for r in mine])
        warn = sum(r["warnings"] for r in mine)
        lines.append(
            f"| {arm} | {m:.1f} | {b2m:.1f} | {b0m:.1f} | {better}/{len(mine)} "
            f"| {sd:.1f} | {warn} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(decision, indent=1, default=float),
        "```",
        "",
    ]
    return "\n".join(lines)
