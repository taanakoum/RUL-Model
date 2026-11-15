"""Gate 3: calibration of the seed ensemble, the causal filter, and the design freeze
(PROJECT_PLAN.md §9, §17.3 decisions 9-10, D15; part P15).

**Data.** For every outer fold (validation tool V, test tool T) the study holds

* the outer fits (3 seeds, trained on the 8 training tools): their predictions on V;
* the inner fits (``inner.n`` splits of the 8 training tools × 3 seeds): each trained on the
  other training tools and predicting its held-out tools. Their predictions are the
  **out-of-fold** (OOF) data the calibration is fitted on.

Only rows with role ``validation`` or ``heldout`` are ever read; the test tool is neither built
nor predicted by an inner fit, and an outer fit's test rows are filtered out when read (D15).

**Methods** (one 90 % central interval per prediction point, window points only):

* ``raw``: the mixture as the seeds give it (reported only);
* ``a``: one scale on every member's σ, fitted on V itself, so its score on V is *in sample*
  (reported only: it cannot be compared fairly, see ``decision.candidates``);
* ``b``: the same scale fitted on the OOF window points of the 8 training tools;
* ``c``: cross-conformal: the 90 % quantile of |y − μ|/sd over the same OOF points, interval
  μ ± q·sd.

**Choice** (Rule B): among the candidates whose mean validation coverage over the folds lies
in the band, the lowest mean Winkler score; candidates within half the relative threshold of it
are tied and the first in ``candidates`` order wins (b before c: b keeps Student-t quantiles).

**Filter** (Rule A): a Kalman filter per tool whose measurement is the ensemble mean with the
b-calibrated variance and whose q is fitted on the OOF tools; the filtered interval
x ± k·sqrt(P) uses k from the OOF standardized errors (a conformal quantile, like c). It is
adopted if its Winkler score beats the chosen method's by the relative threshold on at least
7 of the 10 validation tools and its coverage is in the band.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rulfuse.core.folds import Fold
from rulfuse.core.store import SegmentStore, stable_hash
from rulfuse.inference import CAP_MARGIN_FRACTION
from rulfuse.metrics import winkler
from rulfuse.stats import rule_a, rule_b
from rulfuse.uncertainty import (
    conformal_interval,
    conformal_scale,
    fit_q,
    fit_scale,
    interval,
    kalman_rul,
    mixture_cdf,
    mixture_moments,
)

ROLES = ("validation", "heldout")  # the only roles this module reads (D15)
RELIABILITY_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
METHODS = ("raw", "a", "b", "c")


def read_role_rows(path: Path, role: str) -> dict[str, np.ndarray]:
    """The rows of one role of a predictions file; test rows are never loaded (D15)."""
    if role not in ROLES:
        raise ValueError(f"gate 3 reads roles {ROLES} only, not {role!r}")
    import pyarrow.parquet as pq

    table = pq.read_table(path, filters=[("role", "=", role)])
    return {c: table.column(c).to_numpy(zero_copy_only=False) for c in table.column_names}


def stack_seeds(parts: list[dict[str, np.ndarray]]) -> dict:
    """Seed members of one tool: rows put in time order, then stacked to (S, N)."""
    parts = [{k: v[np.argsort(p["tau_s"], kind="stable")] for k, v in p.items()} for p in parts]
    first = parts[0]
    for p in parts[1:]:
        if not np.array_equal(p["tau_s"], first["tau_s"]):
            raise ValueError("seed members disagree on the prediction points of a tool")
    return {
        "tau": first["tau_s"],
        "rul": first["rul_s"],
        "window": first["window"].astype(bool),
        "mu": np.stack([p["mu_s"] for p in parts]),
        "sigma": np.stack([p["sigma_s"] for p in parts]),
        "nu": np.stack([p["nu"] for p in parts]),
    }


def outer_validation(study, arm: str, fold: Fold, runs: Path | None) -> dict | None:
    from rulfuse.study import fit_dir

    parts = []
    for seed in study.arm_seeds(arm):
        folder = fit_dir(study.run_dir(runs), arm, fold, seed)
        if not (folder / "DONE").is_file():
            return None
        parts.append(read_role_rows(folder / "predictions.parquet", "validation"))
    return stack_seeds(parts)


def fold_code_versions(study, arm: str, fold: Fold, runs: Path | None) -> list[int]:
    """The code versions of the fold's outer and inner fits, and a check that every inner fit
    trained for exactly its outer fit's kept step (§7.2 step 10)."""
    from rulfuse.study import fit_dir, inner_fit_dir, inner_split

    versions = set()
    for seed in study.arm_seeds(arm):
        outer = json.loads(
            (fit_dir(study.run_dir(runs), arm, fold, seed) / "selection.json").read_text(
                encoding="utf-8"
            )
        )
        versions.add(int(outer.get("code_version", 1)))
        for k in range(len(inner_split(fold, **study.inner))):
            folder = inner_fit_dir(study.run_dir(runs), arm, fold, k, seed)
            info = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
            if info.get("selected_step") != outer.get("selected_step"):
                raise ValueError(
                    f"{folder}: trained {info.get('selected_step')} steps but its outer fit "
                    f"kept step {outer.get('selected_step')}"
                )
            versions.add(int(info.get("code_version", 1)))
    return sorted(versions)


def oof_tools(study, arm: str, fold: Fold, runs: Path | None) -> list[dict] | None:
    """The out-of-fold tools of one fold: every training tool, predicted by the inner fits that
    held it out (seeds stacked). None if an inner fit is not finished."""
    from rulfuse.study import inner_fit_dir, inner_split

    spec = study.inner
    tools = []
    for k, (held, _fit_units) in enumerate(inner_split(fold, **spec)):
        by_seed = []
        for seed in study.arm_seeds(arm):
            folder = inner_fit_dir(study.run_dir(runs), arm, fold, k, seed)
            if not (folder / "DONE").is_file():
                return None
            by_seed.append(read_role_rows(folder / "predictions.parquet", "heldout"))
        for unit in held:
            tool = stack_seeds([{c: v[p["unit"] == unit] for c, v in p.items()} for p in by_seed])
            tool["unit"] = unit
            tools.append(tool)
    return tools


def _window(tool: dict) -> dict:
    w = tool["window"]
    return {
        "y": tool["rul"][w],
        "mu": tool["mu"][:, w],
        "sigma": tool["sigma"][:, w],
        "nu": tool["nu"][:, w],
    }


def pool(tools: list[dict]) -> dict:
    """Window points of several tools, concatenated along the point axis."""
    parts = [_window(t) for t in tools]
    return {
        "y": np.concatenate([p["y"] for p in parts]),
        **{k: np.concatenate([p[k] for p in parts], axis=1) for k in ("mu", "sigma", "nu")},
    }


def _interval_stats(y, lo, hi, alpha: float = 0.1) -> dict:
    return {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "winkler": winkler(y, lo, hi, alpha),
        "width": float(np.mean(hi - lo)),
    }


def _pit_inside(y, mu, sigma, nu, level: float) -> np.ndarray:
    alpha = 1.0 - level
    pit = mixture_cdf(y, mu, sigma, nu)
    return (pit >= alpha / 2) & (pit <= 1 - alpha / 2)


def fold_calibration(
    study, arm: str, fold: Fold, store: SegmentStore, runs: Path | None, level: float = 0.9
) -> dict:
    """Everything Gate 3 needs from one fold; ``{"done": False}`` if a fit is missing."""
    horizon = store.settings.horizon_s
    val = outer_validation(study, arm, fold, runs)
    oof = oof_tools(study, arm, fold, runs)
    if val is None or oof is None:
        return {"fold_test": fold.test, "validation": fold.validation, "done": False}
    alpha = 1.0 - level
    pooled = pool(oof)
    y_o, mu_o, sg_o, nu_o = pooled["y"], pooled["mu"], pooled["sigma"], pooled["nu"]
    vw = _window(val)
    y_v, mu_v, sg_v, nu_v = vw["y"], vw["mu"], vw["sigma"], vw["nu"]
    scales = {
        "raw": 1.0,
        "a": fit_scale(y_v, mu_v, sg_v, nu_v, level, horizon),
        "b": fit_scale(y_o, mu_o, sg_o, nu_o, level, horizon),
    }
    mix_o, var_o = mixture_moments(mu_o, sg_o, nu_o)
    q_c = conformal_scale(y_o, mix_o, np.sqrt(var_o), level)
    out: dict = {
        "fold_test": fold.test,
        "validation": fold.validation,
        "done": True,
        "points": int(len(y_v)),
        "oof_points": int(len(y_o)),
        "scale_a": scales["a"],
        "scale_b": scales["b"],
        "q_c": q_c,
        "code_versions": fold_code_versions(study, arm, fold, runs),
    }
    mix_v, var_v = mixture_moments(mu_v, sg_v, nu_v)
    methods = {}
    for m in ("raw", "a", "b"):
        lo, hi = interval(level, mu_v, sg_v * scales[m], nu_v, horizon)
        methods[m] = _interval_stats(y_v, lo, hi, alpha)
    if np.isfinite(q_c):
        lo, hi = conformal_interval(mix_v, np.sqrt(var_v), q_c, horizon)
    else:
        lo, hi = np.zeros_like(y_v), np.full_like(y_v, horizon)
    methods["c"] = _interval_stats(y_v, lo, hi, alpha)
    out["methods"] = methods
    out["window_mae_s"] = float(np.mean(np.abs(mix_v - y_v)))
    # reliability on V: b by the mixture PIT, c by OOF quantiles of the normalized error
    score_o = np.abs(y_o - mix_o) / np.sqrt(var_o)
    rel_b, rel_c = [], []
    for lv in RELIABILITY_LEVELS:
        c_lv = fit_scale(y_o, mu_o, sg_o, nu_o, lv, horizon)
        rel_b.append(float(np.mean(_pit_inside(y_v, mu_v, sg_v * c_lv, nu_v, lv))))
        q_lv = conformal_scale(y_o, mix_o, np.sqrt(var_o), lv)
        rel_c.append(float(np.mean(np.abs(y_v - mix_v) <= q_lv * np.sqrt(var_v))))
    out["reliability"] = {"levels": list(RELIABILITY_LEVELS), "b": rel_b, "c": rel_c}
    out["oof_score_median"] = float(np.median(score_o))
    out["filter"] = _filter_fold(val, oof, scales["b"], horizon, level)
    out["filter"]["unfiltered_window_mae_s"] = out["window_mae_s"]
    return out


def _calibrated_series(tool: dict, c: float):
    """Whole-tool mixture mean and the variance with every σ scaled by c."""
    mean, var = mixture_moments(tool["mu"], tool["sigma"] * c, tool["nu"])
    return mean, var


def _filter_fold(val: dict, oof: list[dict], c_b: float, horizon: float, level: float) -> dict:
    margin = horizon * CAP_MARGIN_FRACTION
    units = []
    for t in oof:
        mean, var = _calibrated_series(t, c_b)
        units.append((t["tau"], t["rul"], mean, var, t["window"]))
    q = fit_q(units, horizon_s=horizon, cap_margin_s=margin)
    z_o = []
    for tau, y, mean, var, w in units:
        x, p = kalman_rul(tau, mean, var, q, horizon, margin)
        z_o.append((np.abs(y - x) / np.sqrt(np.maximum(p, 1e-12)))[w])
    k = conformal_scale(np.concatenate(z_o), np.zeros(sum(len(z) for z in z_o)), np.ones(1), level)
    mean_v, var_v = _calibrated_series(val, c_b)
    x, p = kalman_rul(val["tau"], mean_v, var_v, q, horizon, margin)
    w = val["window"]
    sd = np.sqrt(np.maximum(p, 1e-12))
    lo, hi = conformal_interval(x, sd, k, horizon)
    stats = _interval_stats(val["rul"][w], lo[w], hi[w], 1.0 - level)
    stats["q"] = q
    stats["k"] = k
    stats["window_mae_s"] = float(np.mean(np.abs(x[w] - val["rul"][w])))
    return stats


# ----------------------------------------------------------------------------- decision


def decide_calibration(study, cells: list[dict]) -> dict:
    """Gate 3's two decisions from the per-fold cells (see the module docstring)."""
    rule = study.spec["decision"]
    lo_band, hi_band = (float(v) for v in rule.get("coverage_band", [0.85, 0.95]))
    rel = float(rule.get("threshold_rel", 0.03))
    candidates = list(rule.get("candidates", ["b", "c"]))
    if not all(c.get("done") for c in cells):
        missing = sorted(str(c["fold_test"]) for c in cells if not c.get("done"))
        return {"complete": False, "missing": missing, "adopted": {}}
    cells = sorted(cells, key=lambda c: c["validation"])
    versions = sorted({v for c in cells for v in c.get("code_versions", [1])})
    cov = {m: np.array([c["methods"][m]["coverage"] for c in cells]) for m in METHODS}
    wink = {m: np.array([c["methods"][m]["winkler"] for c in cells]) for m in METHODS}
    in_band = {m: bool(lo_band <= cov[m].mean() <= hi_band) for m in METHODS}
    # reported only (D28): the same means weighted by window points, and without the validation
    # tool with the fewest window points (T4 on Nonastreda), whose few points weigh as much as
    # any other tool in the per-tool mean
    pts = np.array([c["points"] for c in cells], float)
    w = pts / pts.sum()
    thin = int(np.argmin(pts))
    rest = np.arange(len(cells)) != thin
    if not rest.any():  # a single fold (toy runs): nothing to leave out
        rest[:] = True
    out: dict = {
        "complete": True,
        "type": "calibration",
        "code_versions": versions,
        "mixed_code_versions": len(versions) > 1,
        "threshold_rel": rel,
        "coverage_band": [lo_band, hi_band],
        "candidates": candidates,
        "mean_winkler": {m: float(wink[m].mean()) for m in METHODS},
        "mean_coverage": {m: float(cov[m].mean()) for m in METHODS},
        "per_tool_coverage": {m: [float(v) for v in cov[m]] for m in METHODS},
        "in_band": in_band,
        "pooled": {
            m: {"coverage": float((cov[m] * w).sum()), "winkler": float((wink[m] * w).sum())}
            for m in METHODS
        },
        "without_thinnest": {
            "tool": cells[thin]["validation"],
            "points": int(pts[thin]),
            **{
                m: {"coverage": float(cov[m][rest].mean()), "winkler": float(wink[m][rest].mean())}
                for m in METHODS
            },
        },
        "means": {m: float(wink[m].mean()) for m in METHODS},
    }
    feasible = [m for m in candidates if in_band[m]]
    out["constraint_met"] = bool(feasible)
    pick_from = feasible or sorted(candidates, key=lambda m: abs(cov[m].mean() - 0.9))[:1]
    best = min(float(wink[m].mean()) for m in pick_from)
    res = rule_b(
        {m: wink[m] for m in pick_from},
        threshold_s=rel * best,
        cost={m: float(candidates.index(m)) for m in pick_from},
    )
    chosen = res["chosen"]
    out["rule_b_calibration"] = res
    filt = {
        "winkler": np.array([c["filter"]["winkler"] for c in cells]),
        "coverage": np.array([c["filter"]["coverage"] for c in cells]),
        "mae": np.array([c["filter"]["window_mae_s"] for c in cells]),
        "unfiltered_mae": np.array([c["filter"]["unfiltered_window_mae_s"] for c in cells]),
    }
    base = wink[chosen]
    ra = rule_a(base, filt["winkler"], threshold_s=rel * float(base.mean()))
    f_in_band = bool(lo_band <= filt["coverage"].mean() <= hi_band)
    out["rule_a_filter"] = ra
    out["filter"] = {
        "mean_winkler": float(filt["winkler"].mean()),
        "mean_coverage": float(filt["coverage"].mean()),
        "mean_window_mae_s": float(filt["mae"].mean()),
        "unfiltered_mean_window_mae_s": float(filt["unfiltered_mae"].mean()),
        "in_band": f_in_band,
        "adopted": bool(ra["adopted"] and f_in_band),
    }
    rel_curve = {
        m: np.mean([c["reliability"][m] for c in cells], axis=0).tolist() for m in ("b", "c")
    }
    out["reliability"] = {"levels": list(RELIABILITY_LEVELS), **rel_curve}
    out["calibration_choice"] = {
        "method": chosen,
        "level": float(rule.get("level", 0.9)),
        "filter": out["filter"]["adopted"],
    }
    out["new_adoptions"] = {}  # nothing here changes a trained fit; the choice is above
    return out


def render_calibration_report(study, cells: list[dict], decision: dict) -> str:
    lines = [f"# Study {study.name} (Gate 3)", ""]
    lines += [
        "Validation tools only (D15). 90 % central intervals on window points; the scale of "
        "b and the quantile of c come from the out-of-fold training tools, never from the "
        "validation tool. Method a is fitted on the validation tool, so its score is in sample.",
        "",
    ]
    if not decision.get("complete"):
        return "\n".join(lines + [f"Incomplete: missing folds {decision.get('missing')}", ""])
    if decision.get("mixed_code_versions"):
        lines += [
            f"**Warning: fits of code versions {decision['code_versions']} are mixed** "
            "(outer and inner fits of different code): the calibration is not like for like.",
            "",
        ]
    lines += [
        "| Method | Mean coverage | Mean Winkler (s) | In band | Candidate |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        lines.append(
            f"| {m} | {decision['mean_coverage'][m]:.3f} | {decision['mean_winkler'][m]:.1f} "
            f"| {decision['in_band'][m]} | {m in decision['candidates']} |"
        )
    pooled, thin = decision["pooled"], decision["without_thinnest"]
    lines += [
        "",
        "Reported only (D28): the same, weighted by window points, and without the validation "
        f"tool with the fewest points ({thin['tool']}, {thin['points']} points).",
        "",
        "| Method | Pooled coverage | Pooled Winkler (s) | Coverage w/o it | Winkler w/o it |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        lines.append(
            f"| {m} | {pooled[m]['coverage']:.3f} | {pooled[m]['winkler']:.1f} "
            f"| {thin[m]['coverage']:.3f} | {thin[m]['winkler']:.1f} |"
        )
    f = decision["filter"]
    lines += [
        "",
        f"Chosen calibration: **{decision['calibration_choice']['method']}** "
        f"(constraint met: {decision['constraint_met']}).",
        "",
        "## Causal filter",
        "",
        f"Winkler {f['mean_winkler']:.1f} s, coverage {f['mean_coverage']:.3f}, window MAE "
        f"{f['mean_window_mae_s']:.1f} s (unfiltered {f['unfiltered_mean_window_mae_s']:.1f} s); "
        f"Rule A gain {decision['rule_a_filter']['gain_mean']:.1f} s, "
        f"{decision['rule_a_filter']['wins']}/{decision['rule_a_filter']['n']} wins; "
        f"adopted: **{f['adopted']}**.",
        "",
        "## Per validation tool",
        "",
        "| Validation tool | Points | Scale a | Scale b | q (c) | Cov raw | Cov a | Cov b | Cov c "
        "| Winkler b | Winkler c |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(cells, key=lambda c: c["validation"]):
        m = c["methods"]
        lines.append(
            f"| {c['validation']} | {c['points']} | {c['scale_a']:.2f} | {c['scale_b']:.2f} "
            f"| {c['q_c']:.2f} | {m['raw']['coverage']:.2f} | {m['a']['coverage']:.2f} "
            f"| {m['b']['coverage']:.2f} | {m['c']['coverage']:.2f} "
            f"| {m['b']['winkler']:.0f} | {m['c']['winkler']:.0f} |"
        )
    rel = decision["reliability"]
    lines += [
        "",
        "## Reliability (nominal level → coverage on the validation tools)",
        "",
        "| Level | " + " | ".join(f"{v:.2f}" for v in rel["levels"]) + " |",
        "|---|" + "---|" * len(rel["levels"]),
        "| b | " + " | ".join(f"{v:.2f}" for v in rel["b"]) + " |",
        "| c | " + " | ".join(f"{v:.2f}" for v in rel["c"]) + " |",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(decision, indent=1, default=_json),
        "```",
        "",
    ]
    return "\n".join(lines)


def _json(x):
    from rulfuse.study import _json_value

    return _json_value(x)


def aggregate_calibration(study, runs: Path | None, store: SegmentStore) -> tuple[dict, list]:
    """Cells of every fold and the decision; the arm is the study's single arm."""
    from rulfuse.study import study_folds

    arm = study.spec["decision"].get("arm") or next(iter(study.arms))
    level = float(study.spec["decision"].get("level", 0.9))
    cells = [
        fold_calibration(study, arm, fold, store, runs, level)
        for fold in study_folds(study, store.unit_names)
    ]
    return decide_calibration(study, cells), cells


# ----------------------------------------------------------------------------- freeze


def freeze(
    study, runs: Path | None, out_path: Path, date: str | None = None, amended: str | None = None
) -> dict:
    """Write the frozen design (D15): the full configuration, the calibration choice and the
    filter, with a hash. Refuses an incomplete Gate 3 and one whose coverage constraint failed:
    finishing the computation is not passing the gate. ``amended`` is the text of a recorded
    decision that accepts the failure; it is stored in the file. Nothing is trained or scored."""
    decision = json.loads((study.run_dir(runs) / "decision.json").read_text(encoding="utf-8"))
    if not decision.get("complete"):
        raise ValueError("gate 3 is incomplete: aggregate it first")
    if not decision.get("constraint_met") and not amended:
        raise ValueError(
            "gate 3 failed its coverage constraint (no candidate has mean validation coverage "
            f"in {decision['coverage_band']}); record an amended decision before freezing"
        )
    arm = study.spec["decision"].get("arm") or next(iter(study.arms))
    frozen = {
        "frozen_after": study.name,
        "design": study.config(arm, runs).to_dict(),
        "store": study.spec["store"],
        "signal_encoder": study.spec["signal_encoder"],
        "image_encoder": study.spec["image_encoder"],
        "seeds": study.seeds,
        "inner": study.inner,
        "calibration": decision["calibration_choice"],
        "coverage_band": decision["coverage_band"],
        "constraint_met": bool(decision["constraint_met"]),
        "amended": amended,
        "code_versions": decision.get("code_versions"),
    }
    frozen["hash"] = stable_hash(frozen)
    from rulfuse.study import FREEZE_DATE

    frozen["freeze_date"] = date or FREEZE_DATE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    return frozen
