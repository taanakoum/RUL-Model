"""The report of B6's single full run (docs/OMNIFUSER_RUL_PLAN.md version 4, §7).

One configuration (force + the admitted photo types + usage; the GRU memory and the
temporal loss on), 10 folds × 3 seeds. ``write_run_report`` reads the run's fits and writes
``<runs root>/<dataset>-<store>/<arm>/report/``:

* ``REPORT.md``: the run, the test results (each tool once, as the test tool of its fold),
  the paired comparison with B2, the error by RUL range, the validation results;
* ``run_summary.json``: every number of the report;
* ``test_per_unit.csv``, ``validation_per_unit.csv``, ``error_by_bin.csv``,
  ``test_trajectories.csv``, ``curves.csv``;
* ``figures/``: the RUL trajectories of the test tools, the per-tool errors and the error
  by RUL range (when matplotlib is installed).

Everything is scored on the evaluation grid's window points (the project's 10 s points with
RUL ≤ H), the same points as RUL-Fuse and B0-B2, with each point predicted by the mean of
the seeds. Test results are included on or after the design freeze (D15), or before it
with ``unseal``, which the report records.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from rulfuse import metrics as M
from rulfuse.baselines import b0, b1, b2, b2_interval
from rulfuse.core.folds import Fold, leave_one_unit_out
from rulfuse.core.store import SegmentStore
from rulfuse.omnifuser_rul import aggregate
from rulfuse.omnifuser_rul.report import _curve_rows, _write_csv, arm_progress
from rulfuse.omnifuser_rul.train import ArmSpec, arm_dir

BIN_S = 300.0
LATE_FRACTION = 1.0 / 15.0  # PROJECT_PLAN.md §10.2: late = more than H/15 too optimistic
KEYS = (
    "window_mae",
    "window_rmse",
    "window_bias",
    "whole_life_mae",
    "late_rate",
    "early_rate",
    "phm_score",
    "relative_accuracy",
    "increase_rate",
    "mean_increase_s",
)


def test_open(unseal: bool, today: date | None = None) -> bool:
    return unseal or (today or date.today()) >= aggregate.FREEZE


def _fold_of(folds: list[Fold], unit: str, role: str) -> Fold:
    return next(f for f in folds if (f.test if role == "test" else f.validation) == unit)


def _unit_results(
    store: SegmentStore, preds: dict, combined: dict, unit: str, fold: Fold, role: str
):
    seg, horizon = store.segments, store.settings.horizon_s
    mine = combined["unit"] == unit
    rows = combined["row"][mine]
    order = np.argsort(seg["tau_s"][rows].astype(float), kind="stable")
    rows = rows[order]
    pred = combined["prediction_s"][mine][order]
    grid = combined["on_grid"][mine][order].astype(bool)
    tau = seg["tau_s"][rows].astype(float)
    rul, y = seg["rul_s"][rows].astype(float), seg["y_s"][rows].astype(float)
    window = seg["window"][rows].astype(bool) & grid
    failures = np.array([store.units[u]["failure_s"] for u in fold.train], float)
    late = horizon * LATE_FRACTION
    yw, pw, tw = rul[window], pred[window], tau[window]
    base = {
        "B0": b0(len(yw), horizon),
        "B1": b1(failures, tw, horizon),
        "B2": b2(failures, tw, horizon),
    }
    lower, upper = b2_interval(failures, tw, horizon)
    uncapped = y < horizon - 1e-6
    stability = M.temporal_stability(pred, tau, uncapped)
    # per seed, on the same window points
    keep = (preds["role"] == role) & (preds["unit"] == unit)
    by_seed: dict[int, dict[int, float]] = {}
    for r, s, v in zip(
        preds["row"][keep], preds["seed"][keep], preds["prediction_s"][keep], strict=True
    ):
        by_seed.setdefault(int(s), {})[int(r)] = float(v)
    window_rows = rows[window]
    seed_mae = {
        s: M.mae(yw, np.array([values[int(r)] for r in window_rows]))
        for s, values in sorted(by_seed.items())
    }
    grid_rows = rows[grid]
    per_row = np.array(
        [[values.get(int(r), np.nan) for values in by_seed.values()] for r in grid_rows]
    )
    row = {
        "unit": unit,
        "fold_test": fold.test,
        "points_window": int(window.sum()),
        "points_grid": int(grid.sum()),
        "points_all": int(len(rows)),
        "window_mae": M.mae(yw, pw),
        "window_rmse": M.rmse(yw, pw),
        "window_bias": M.bias(yw, pw),
        "whole_life_mae": M.mae(y[grid], pred[grid]),
        "late_rate": M.late_rate(yw, pw, late),
        "early_rate": M.early_rate(yw, pw, late),
        "phm_score": M.phm_score(yw, pw, horizon),
        "relative_accuracy": M.relative_accuracy(yw, pw),
        **stability,
        "seed_window_mae": seed_mae,
        "seed_sd": float(np.std(list(seed_mae.values()), ddof=1)) if len(seed_mae) > 1 else 0.0,
        **{f"{name}_window_mae": M.mae(yw, p) for name, p in base.items()},
        "B2_coverage90": M.coverage(yw, lower, upper),
        "B2_phm_score": M.phm_score(yw, base["B2"], horizon),
        "B2_late_rate": M.late_rate(yw, base["B2"], late),
    }
    trajectory = {
        "unit": unit,
        "tau_s": tau[grid],
        "y_s": y[grid],
        "b6_s": pred[grid],
        "seed_min_s": np.nanmin(per_row, axis=1),
        "seed_max_s": np.nanmax(per_row, axis=1),
        "b2_s": b2(failures, tau[grid], horizon),
        "failure_s": float(store.units[unit]["failure_s"]),
    }
    pooled = {"y": yw, "b6": pw, "b2": base["B2"]}
    return row, trajectory, pooled


def _summarize(rows: list[dict], pooled: list[dict], horizon: float) -> dict:
    summary: dict = {}
    for key in (
        *KEYS,
        "B0_window_mae",
        "B1_window_mae",
        "B2_window_mae",
        "B2_phm_score",
        "B2_late_rate",
    ):
        values = np.array([r[key] for r in rows], float)
        lo, hi = M.bootstrap_mean_ci(values)
        summary[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95": [lo, hi],
        }
    y = np.concatenate([p["y"] for p in pooled])
    b6, base = np.concatenate([p["b6"] for p in pooled]), np.concatenate([p["b2"] for p in pooled])
    summary["pooled"] = {
        "points": int(len(y)),
        "window_mae": M.mae(y, b6),
        "window_rmse": M.rmse(y, b6),
        "B2_window_mae": M.mae(y, base),
    }
    gain = np.array([r["B2_window_mae"] - r["window_mae"] for r in rows])
    from scipy.stats import wilcoxon

    p = float(wilcoxon(gain, alternative="greater").pvalue) if np.any(gain != 0) else 1.0  # pyright: ignore[reportAttributeAccessIssue]
    lo, hi = M.bootstrap_mean_ci(gain)
    summary["vs_B2"] = {
        "mean_gain_s": float(gain.mean()),
        "ci95": [lo, hi],
        "units_better": int((gain > 0).sum()),
        "units": int(len(gain)),
        "p_one_sided_wilcoxon": p,
    }
    edges = np.arange(0.0, horizon + BIN_S, BIN_S)
    summary["error_by_bin"] = {
        "B6": M.error_by_bin(y, b6, edges),
        "B2": M.error_by_bin(y, base, edges),
    }
    return summary


def _role_results(store, preds, role):
    folds = leave_one_unit_out(store.unit_names)
    combined = aggregate.ensemble(preds, role)
    rows, trajectories, pooled = [], [], []
    for unit in store.unit_names:
        if not (combined["unit"] == unit).any():
            continue
        r, t, p = _unit_results(store, preds, combined, unit, _fold_of(folds, unit, role), role)
        rows.append(r)
        trajectories.append(t)
        pooled.append(p)
    return rows, trajectories, _summarize(rows, pooled, store.settings.horizon_s)


def _fmt(v, digits=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{digits}f}"


def _ci(s: dict, digits: int = 1) -> str:
    lo, hi = (_fmt(x, digits) for x in s["ci95"])
    return f"{_fmt(s['mean'], digits)} ± {_fmt(s['sd'], digits)} ({lo} to {hi})"


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _run_lines(res: dict) -> list[str]:
    spec, prog = res["spec"], res["progress"]
    epochs = prog["selected_epoch"]
    seeds = ", ".join(str(s) for s in spec["seeds"])
    missing = f"; missing {', '.join(prog['missing'][:6])}" if prog["missing"] else ""
    return [
        "# B6 OmniFuser-RUL: results of the full run",
        "",
        f"Store `{res['store']}` (B6's 1 s store); written {res['written']}.",
        "",
        "## The run",
        "",
        f"- **Configuration:** players {', '.join(spec['players'])} and usage; GRU memory "
        f"(hidden {spec['memory_hidden']}, blocks of {spec['memory_block']} steps); loss = "
        f"point-weighted MSE + {spec['mono_weight']:g} × monotonic penalty (a predicted RUL "
        f"that rises) + {spec['slope_weight']:g} × slope consistency; {spec['epochs']} epochs "
        f"of {spec['pairs_per_epoch']} pairs; seeds {seeds}.",
        f"- **Fits:** {prog['done']} of {prog['expected']}{missing}; median "
        f"{_fmt(prog['fit_seconds']['median'], 0)} s per fit; selected epoch median "
        f"{_fmt(epochs['median'], 0)} (range {epochs['min']}-{epochs['max']}); "
        f"{epochs['best_triple_at_end']} fits were still improving at the last epoch.",
        "- **Points:** each fold's test tool is scored on the project's 10 s points with "
        "RUL ≤ H = 1,500 s (the same points as RUL-Fuse and B0-B2). B6 predicts every 1 s "
        "point; each point's prediction is the mean of the seeds.",
        "",
    ]


def _test_lines(res: dict) -> list[str]:
    if "test" not in res:
        return [
            "## Test results",
            "",
            f"Sealed until {aggregate.FREEZE} (D15). Every fit has written its test",
            "predictions; rerun the report on or after that date, or with `--unseal` to read",
            "them earlier.",
            "",
        ]
    t, lines = res["test"]["summary"], []
    if res["test"].get("unsealed_before_freeze"):
        lines += [
            f"> **Test read on {res['test']['read_on']}, before the design freeze "
            f"({aggregate.FREEZE}), with `--unseal`.** B6 is a stand-alone baseline; no "
            "RUL-Fuse design decision may use these numbers before the freeze (D15).",
            "",
        ]
    lines += [
        "## Test results (each of the 10 tools as the test tool of its fold)",
        "",
        "Mean ± sd over the test tools (95 % bootstrap interval of the mean over tools).",
        "",
        _row("Metric", "B6", "B2 (usage rule)"),
        "|---|---|---|",
        _row("Window MAE (s)", _ci(t["window_mae"]), _ci(t["B2_window_mae"])),
        _row("Window RMSE (s)", _ci(t["window_rmse"]), "–"),
        _row("Bias (s; positive = too optimistic)", _ci(t["window_bias"]), "–"),
        _row("Whole-life MAE of the capped target (s)", _ci(t["whole_life_mae"]), "–"),
        _row(
            "Late predictions (> 100 s too optimistic)",
            _ci(t["late_rate"], 3),
            _ci(t["B2_late_rate"], 3),
        ),
        _row("Early predictions (> 100 s too pessimistic)", _ci(t["early_rate"], 3), "–"),
        _row(
            "Asymmetric score (§10.2; lower is better)",
            _ci(t["phm_score"], 3),
            _ci(t["B2_phm_score"], 3),
        ),
        _row("Within ±20 % of the true RUL", _ci(t["relative_accuracy"], 3), "–"),
        _row("Rising steps (share of 1 s steps where ŷ rises)", _ci(t["increase_rate"], 3), "0"),
        "",
        f"Pooled over all {t['pooled']['points']} window points: B6 MAE "
        f"{_fmt(t['pooled']['window_mae'])} s, B2 {_fmt(t['pooled']['B2_window_mae'])} s. "
        "Other baselines (mean window MAE): B0, the constant H/2 (not deployable), "
        f"{_fmt(t['B0_window_mae']['mean'])} s; B1 {_fmt(t['B1_window_mae']['mean'])} s.",
        "",
        "### Against B2, paired over the test tools",
        "",
    ]
    v = t["vs_B2"]
    lines += [
        f"B6 lowers the window MAE by **{_fmt(v['mean_gain_s'])} s** on average (95 % "
        f"interval {_fmt(v['ci95'][0])} to {_fmt(v['ci95'][1])} s), better on "
        f"{v['units_better']} of {v['units']} tools; one-sided Wilcoxon "
        f"p = {v['p_one_sided_wilcoxon']:.3g}.",
        "",
        "### Per test tool",
        "",
        _row("Tool", "Window points", "B6 MAE (s)", "B2 MAE (s)", "Gain (s)", "Seed sd (s)")
        + " Bias (s) | Late | Rising steps |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res["test"]["per_unit"]:
        lines.append(
            _row(
                r["unit"],
                r["points_window"],
                _fmt(r["window_mae"]),
                _fmt(r["B2_window_mae"]),
                _fmt(r["B2_window_mae"] - r["window_mae"]),
                _fmt(r["seed_sd"]),
                _fmt(r["window_bias"]),
                _fmt(r["late_rate"], 2),
                _fmt(r["increase_rate"], 3),
            )
        )
    lines += ["", "### Error by true RUL (pooled window points)", ""]
    lines += [
        _row("True RUL (s)", "Points", "B6 MAE (s)", "B6 bias (s)", "B2 MAE (s)"),
        "|---|---|---|---|---|",
    ]
    for a, b in zip(t["error_by_bin"]["B6"], t["error_by_bin"]["B2"], strict=True):
        span = f"{a['from_s']:.0f}-{a['to_s']:.0f}"
        lines.append(_row(span, a["points"], _fmt(a["mae"]), _fmt(a["bias"]), _fmt(b["mae"])))
    if res.get("figures"):
        lines += ["", "### Figures", ""]
        lines += [f"- `figures/{name}`: {caption}" for name, caption in res["figures"].items()]
    return [*lines, ""]


def _markdown(res: dict) -> str:
    v = res["validation"]["summary"]
    lines = [
        *_run_lines(res),
        *_test_lines(res),
        "## Validation results (each fold's validation tool; used only to select epochs)",
        "",
        f"Window MAE {_ci(v['window_mae'])} s; B2 {_ci(v['B2_window_mae'])} s; better than B2 "
        f"on {v['vs_B2']['units_better']} of {v['vs_B2']['units']} tools.",
        "",
    ]
    return "\n".join(lines)


def write_run_report(
    store: SegmentStore,
    root: Path,
    spec: ArmSpec,
    unseal: bool = False,
    out: Path | None = None,
    today: date | None = None,
    figures: bool = True,
) -> Path:
    folder = arm_dir(root, store, spec)
    out = out or folder / "report"
    out.mkdir(parents=True, exist_ok=True)
    progress = arm_progress(store, root, spec)
    _write_csv(_curve_rows([progress]), out / "curves.csv")
    preds = aggregate.load_predictions(folder, spec.signature())
    today = today or date.today()
    res: dict = {
        "store": f"{store.dataset}/{store.signature}",
        "written": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": json.loads(json.dumps(asdict(spec))),
        "horizon_s": store.settings.horizon_s,
        "progress": {k: v for k, v in progress.items() if k != "_fits"},
    }
    rows, _, summary = _role_results(store, preds, "validation")
    res["validation"] = {"per_unit": rows, "summary": summary}
    _write_csv(_flat(rows), out / "validation_per_unit.csv")
    if test_open(unseal, today):
        rows, trajectories, summary = _role_results(store, preds, "test")
        res["test"] = {
            "per_unit": rows,
            "summary": summary,
            "read_on": today.isoformat(),
            "unsealed_before_freeze": today < aggregate.FREEZE,
        }
        _write_csv(_flat(rows), out / "test_per_unit.csv")
        _write_csv(
            [dict(error="B6", **r) for r in summary["error_by_bin"]["B6"]]
            + [dict(error="B2", **r) for r in summary["error_by_bin"]["B2"]],
            out / "error_by_bin.csv",
        )
        _write_csv(
            [
                {
                    "unit": t["unit"],
                    "tau_s": a,
                    "y_s": b,
                    "b6_s": c,
                    "seed_min_s": d,
                    "seed_max_s": e,
                    "b2_s": f,
                }
                for t in trajectories
                for a, b, c, d, e, f in zip(
                    t["tau_s"],
                    t["y_s"],
                    t["b6_s"],
                    t["seed_min_s"],
                    t["seed_max_s"],
                    t["b2_s"],
                    strict=True,
                )
            ],
            out / "test_trajectories.csv",
        )
        if figures:
            res["figures"] = make_figures(res, trajectories, out / "figures")
    (out / "run_summary.json").write_text(
        json.dumps(res, indent=1, default=float), encoding="utf-8"
    )
    (out / "REPORT.md").write_text(_markdown(res), encoding="utf-8")
    return out


def _flat(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        flat = {k: v for k, v in r.items() if k != "seed_window_mae"}
        flat.update({f"seed{s}_window_mae": v for s, v in r["seed_window_mae"].items()})
        out.append(flat)
    return out


# ----------------------------------------------------------------------------- figures

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
B6_COLOR, B2_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def _style(ax, grid: str = "both") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8)


def make_figures(res: dict, trajectories: list[dict], folder: Path) -> dict[str, str]:
    """The report's figures (light theme, the reference palette); {} without matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}
    folder.mkdir(parents=True, exist_ok=True)
    horizon = float(res["horizon_s"])
    made: dict[str, str] = {}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK})

    # 1. RUL trajectories of the test tools (small multiples, one shared legend)
    n = len(trajectories)
    cols = 5 if n > 4 else n
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(2.6 * cols, 2.3 * rows_n), squeeze=False, sharey=True
    )
    fig.patch.set_facecolor(SURFACE)
    for ax, t in zip(axes.flat, trajectories, strict=False):
        _style(ax)
        x = t["tau_s"] / 60.0
        ax.fill_between(
            x, t["seed_min_s"], t["seed_max_s"], color=B6_COLOR, alpha=0.12, linewidth=0
        )
        ax.plot(x, t["y_s"], color=INK, linewidth=1.6, label="true RUL (capped at H)")
        ax.plot(x, t["b2_s"], color=B2_COLOR, linewidth=1.6, label="B2 usage rule")
        ax.plot(
            x,
            t["b6_s"],
            color=B6_COLOR,
            linewidth=1.6,
            label="B6 (mean of seeds; band: seed range)",
        )
        ax.set_ylim(0, horizon * 1.05)
        ax.set_title(t["unit"], fontsize=9, color=INK, loc="left")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("RUL (s)", color=INK_2)
    for ax in axes[-1]:
        ax.set_xlabel("operating time (min)", color=INK_2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(folder / "rul_trajectories.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    made["rul_trajectories.png"] = "true and predicted RUL of each test tool over its life, with B2"

    # 2. Window MAE per test tool: B6 against B2 (dot plot)
    rows = res["test"]["per_unit"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid="x")  # gridlines along the value axis only
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r["window_mae"], r["B2_window_mae"]], [i, i], color=GRID, linewidth=2, zorder=1)
    ax.scatter(
        [r["B2_window_mae"] for r in rows],
        y,
        s=40,
        color=B2_COLOR,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=2,
        label="B2 usage rule",
    )
    ax.scatter(
        [r["window_mae"] for r in rows],
        y,
        s=40,
        color=B6_COLOR,
        edgecolor=SURFACE,
        linewidth=1.5,
        zorder=3,
        label="B6",
    )
    ax.set_yticks(y, [r["unit"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("window MAE (s)", color=INK_2)
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "window_mae_per_tool.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    made["window_mae_per_tool.png"] = "window MAE of each test tool, B6 and B2"

    # 3. Error by true RUL (grouped columns, 2 px surface gap)
    bins = res["test"]["summary"]["error_by_bin"]
    labels = [f"{b['from_s']:.0f}-{b['to_s']:.0f}" for b in bins["B6"]]
    x = np.arange(len(labels))
    width = 0.15  # about 24 px at this size: thin columns, the rest of the band is air
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax, grid="y")  # gridlines along the value axis only
    ax.bar(x - width / 2 - 0.01, [b["mae"] for b in bins["B6"]], width, color=B6_COLOR, label="B6")
    ax.bar(
        x + width / 2 + 0.01,
        [b["mae"] for b in bins["B2"]],
        width,
        color=B2_COLOR,
        label="B2 usage rule",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel("true RUL (s)", color=INK_2)
    ax.set_ylabel("MAE (s)", color=INK_2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "error_by_rul.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    made["error_by_rul.png"] = "MAE by true-RUL range, pooled over the test tools, B6 and B2"
    return made
