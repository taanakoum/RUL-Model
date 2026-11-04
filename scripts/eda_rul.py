"""Stage 1: data audit and RUL EDA (PROJECT_PLAN.md §17.2, Stage 1).

Writes the numbers to ``<runs>/eda/`` (``eda_summary.json`` and CSV tables) and the figures
to ``docs/eda_rul/``. ``docs/EDA_RUL.md`` explains them.

    D:\\RULModel\\.venv\\Scripts\\python.exe scripts/eda_rul.py

Needs the force store (``rulfuse build-force-store``). About a minute on the laptop; it
refuses to start with less than 800 MB of free memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.signal import welch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from rulfuse import audit, paths  # noqa: E402
from rulfuse.baselines import score_usage_baselines  # noqa: E402
from rulfuse.data.dataset import Dataset, DatasetOptions, load_dataset, trimmed_bounds  # noqa: E402
from rulfuse.data.folds import leave_one_tool_out_folds  # noqa: E402
from rulfuse.data.labels import TTF_RULES, isotonic, tool_labels  # noqa: E402
from rulfuse.data.nonastreda import PHOTO_KINDS, SAMPLE_RATE_HZ, metas_by_tool, photo_path  # noqa: E402
from rulfuse.data.trim import CAUSAL_REFERENCE_SAMPLES, envelope  # noqa: E402
from rulfuse.resources import check_resources  # noqa: E402

OMNIFUSER_RECORDS = (
    paths.REPO_ROOT.parent / "OmniFuser" / "experiment_results" / "eda" / "records.csv"
)
EDA_MD_FLAGGED = ("T2R4B1", "T1R5B1", "T7R1B2", "T9R1B2")  # OmniFuser EDA.md §5
BANDS_HZ = ((0, 50), (50, 150), (150, 300), (300, 500))
FORCE_TITLES = {"rms_F": "RMS of |F|", "rms_Fx": "RMS of Fx", "rms_Fy": "RMS of Fy"}
BASELINE_PASSES = 4

# Reference palette (dataviz skill, references/palette.md), light surface. Slots 1-3 are
# documented as passing every all-pairs gate; this script uses no more than three.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
DPI = 150
LINE = 2 * 72 / DPI  # 2 px in points
MARK = (8 * 72 / DPI) ** 2  # 8 px markers, as scatter area in points^2


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": DPI,
        }
    )


def save(fig: Figure, out: Path) -> None:
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure {out.name}")


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok])[0])  # pyright: ignore[reportArgumentType]


# --------------------------------------------------------------------------- analyses


def duplicate_analysis(ds: Dataset) -> dict:
    patterns = Counter("+".join(f"B{r[-1]}" for r in group) for group in ds.store.groups)
    missing = [
        s for slots in ds.timelines.values() for s in slots if s.recorded and not s.force_present
    ]
    return {
        "groups": len(ds.store.groups),
        "patterns": dict(patterns),
        "force_missing_passes": len(missing),
        "force_missing_share": round(len(missing) / len(ds.metas), 3),
        "force_missing_by_block": dict(sorted(Counter(f"B{s.block}" for s in missing).items())),
        "force_missing_by_tool": {t: sum(1 for s in missing if s.tool == t) for t in ds.tools},
    }


def trim_analysis(ds: Dataset) -> tuple[dict, list[dict]]:
    omni = trimmed_bounds(ds.store, "omnifuser")
    ids = sorted(omni)
    start = np.array([ds.bounds[k][0] - omni[k][0] for k in ids])
    end = np.array([ds.bounds[k][1] - omni[k][1] for k in ids])
    dur = np.array(
        [(ds.bounds[k][1] - ds.bounds[k][0] - omni[k][1] + omni[k][0]) / 1000 for k in ids]
    )
    reproduced = None
    if OMNIFUSER_RECORDS.is_file():
        theirs = {
            r["record_id"]: (int(r["trim_start"]), int(r["trim_end"]))
            for r in csv.DictReader(OMNIFUSER_RECORDS.open())
        }
        reproduced = int(sum(omni[k] == theirs[k] for k in theirs))
    flagged = []
    for k in ids:
        if abs(ds.bounds[k][0] - omni[k][0]) > 200 or k in EDA_MD_FLAGGED:
            force = ds.store.force(k)
            env = envelope(force)
            flagged.append(
                {
                    "record_id": k,
                    "in_eda_md": k in EDA_MD_FLAGGED,
                    "causal_start": ds.bounds[k][0],
                    "omnifuser_start": omni[k][0],
                    "causal_end": ds.bounds[k][1],
                    "omnifuser_end": omni[k][1],
                    "samples": len(force),
                    "causal_reference": round(float(np.median(env[:CAUSAL_REFERENCE_SAMPLES])), 1),
                    "omnifuser_reference": round(
                        float(np.median(env[: int(len(force) * 0.05)])), 1
                    ),
                }
            )
    summary = {
        "passes": len(ids),
        "omnifuser_rule_reproduces_records_csv": reproduced,
        "start_within_200_share": round(float(np.mean(np.abs(start) <= 200)), 4),
        "start_exact_share": round(float(np.mean(start == 0)), 4),
        "start_abs_median": float(np.median(np.abs(start))),
        "start_abs_max": int(np.abs(start).max()),
        "end_exact_share": round(float(np.mean(end == 0)), 4),
        "end_abs_median": float(np.median(np.abs(end))),
        "end_abs_max": int(np.abs(end).max()),
        "duration_diff_s_median": round(float(np.median(dur)), 3),
        "duration_diff_s_abs_max": round(float(np.abs(dur).max()), 3),
        "disagreements_over_200": [
            r["record_id"] for r in flagged if abs(r["causal_start"] - r["omnifuser_start"]) > 200
        ],
        "_start_diff": start.tolist(),
        "_end_diff": end.tolist(),
    }
    return summary, flagged


def duration_analysis(ds: Dataset) -> list[dict]:
    rows = []
    for tool, metas in metas_by_tool(ds.metas).items():
        d = np.array([ds.durations[m.record_id] for m in metas])
        rows.append(
            {
                "tool": tool,
                "passes": len(d),
                "median_s": round(float(np.median(d)), 1),
                "p10_s": round(float(np.quantile(d, 0.1)), 1),
                "p90_s": round(float(np.quantile(d, 0.9)), 1),
                "min_s": round(float(d.min()), 1),
                "max_s": round(float(d.max()), 1),
            }
        )
    return rows


def ttf_analysis(ds: Dataset, ds_rec: Dataset) -> dict:
    out: dict = {}
    for name, data in (("gap", ds), ("recorded", ds_rec)):
        by_tool = metas_by_tool(data.metas)
        out[name] = {}
        for rule in TTF_RULES:
            labels = {t: tool_labels(data.timelines[t], by_tool[t], rule) for t in data.tools}
            positions = [labels[t].ttf_position for t in data.tools]
            out[name][rule] = {
                "ttf_pass": dict(zip(data.tools, positions, strict=True)),
                "mean": round(float(np.mean(positions)), 1),
                "sd": round(float(np.std(positions)), 1),
                "wear_at_ttf_um": {t: round(labels[t].wear_at_ttf_um, 1) for t in data.tools},
                "tau_ttf_s": {t: round(labels[t].tau_ttf) for t in data.tools},
            }
    return out


def window_analysis(ds: Dataset) -> list[dict]:
    rows = []
    for tool, lab in ds.labels.items():
        rows.append(
            {
                "tool": tool,
                "ttf_pass": lab.ttf_position,
                "ttf_record": lab.ttf_record,
                "tau_ttf_s": round(lab.tau_ttf),
                "wear_at_ttf_um": round(lab.wear_at_ttf_um, 1),
                "unrecorded_passes_before_ttf": sum(
                    1
                    for s in ds.timelines[tool]
                    if not s.recorded and s.position < lab.ttf_position
                ),
                "pre_ttf_points": len(lab.points),
                "force_missing_points": int((~lab.points.force_present).sum()),
                "window_points_pass": int(lab.targets.window_pass.sum()),
                "window_points_sec": int(lab.targets.window_sec.sum()),
            }
        )
    return rows


def force_indicators(ds: Dataset) -> tuple[list[dict], dict]:
    """Pass-level force indicators on the causally trimmed force, relative to each tool's
    first recorded passes, against RUL and smoothed wear."""
    rows: list[dict] = []
    by_tool = metas_by_tool(ds.metas)
    for tool, slots in ds.timelines.items():
        lab = ds.labels[tool]
        smoothed = dict(
            zip(
                [m.record_id for m in by_tool[tool]],
                isotonic([m.flank_wear for m in by_tool[tool]]),
                strict=True,
            )
        )
        tool_rows = []
        for slot in slots:
            if slot.record_id is None or slot.force_record is None:
                continue  # unrecorded or force-missing pass
            start, end = ds.bounds[slot.force_record]
            force = np.asarray(ds.store.force(slot.force_record)[start:end], dtype=np.float64)
            fz_rel = force[:, 2] - force[:SAMPLE_RATE_HZ, 2].mean()
            magnitude = np.linalg.norm(force, axis=1)
            freqs, psd = welch(magnitude - magnitude.mean(), fs=SAMPLE_RATE_HZ, nperseg=1024)
            row = {
                "tool": tool,
                "record_id": slot.record_id,
                "position": slot.position,
                "rul_pass_end": lab.ttf_position - slot.position
                if slot.position <= lab.ttf_position
                else None,
                "wear_smoothed_um": smoothed[slot.record_id],
                "rms_Fx": float(np.sqrt(np.mean(force[:, 0] ** 2))),
                "rms_Fy": float(np.sqrt(np.mean(force[:, 1] ** 2))),
                "rms_Fz": float(np.sqrt(np.mean(force[:, 2] ** 2))),
                "rms_Fz_rel": float(np.sqrt(np.mean(fz_rel**2))),
                "rms_F": float(np.sqrt(np.mean(magnitude**2))),
            }
            for lo, hi in BANDS_HZ:
                keep = (freqs > lo) & (freqs <= hi)
                row[f"band_{lo}_{hi}Hz"] = float(np.trapezoid(psd[keep], freqs[keep]))
            tool_rows.append(row)
        indicators = [k for k in tool_rows[0] if k.startswith(("rms_", "band_"))]
        for key in indicators:
            base = np.mean([r[key] for r in tool_rows[:BASELINE_PASSES]])
            for r in tool_rows:
                r[f"rel_{key}"] = r[key] / base - 1.0
        rows.extend(tool_rows)
    indicators = [k for k in rows[0] if k.startswith(("rms_", "band_"))]
    summary = {}
    pre = [r for r in rows if r["rul_pass_end"] is not None]
    for key in indicators:
        per_tool = [
            rho(
                [r[f"rel_{key}"] for r in pre if r["tool"] == t],
                [r["rul_pass_end"] for r in pre if r["tool"] == t],
            )
            for t in ds.tools
        ]
        summary[key] = {
            "rho_rul_pooled_relative": round(
                rho([r[f"rel_{key}"] for r in pre], [r["rul_pass_end"] for r in pre]), 3
            ),
            "rho_rul_pooled_absolute": round(
                rho([r[key] for r in pre], [r["rul_pass_end"] for r in pre]), 3
            ),
            "rho_rul_per_tool_median": round(float(np.nanmedian(per_tool)), 3),
            "rho_rul_per_tool_min": round(float(np.nanmin(per_tool)), 3),
            "rho_wear_pooled_relative": round(
                rho([r[f"rel_{key}"] for r in rows], [r["wear_smoothed_um"] for r in rows]), 3
            ),
            "rho_wear_pooled_absolute": round(
                rho([r[key] for r in rows], [r["wear_smoothed_um"] for r in rows]), 3
            ),
        }
    return rows, summary


def _photo_stats(path: Path, kind: str) -> dict:
    with Image.open(path) as image:
        colour = image.mode not in ("1", "L", "I", "F")  # workpiece photos are greyscale
        if image.format == "JPEG":
            image.draft("RGB", (image.width // 4, image.height // 4))
        rgb = image.convert("RGB")
    if kind == "tool":  # the cutting-edge band (OmniFuser photo probe)
        rgb = rgb.crop((0, round(0.30 * rgb.height), rgb.width, round(0.85 * rgb.height)))
    rgb.thumbnail((400, 400))
    array = np.asarray(rgb, dtype=np.float64) / 255.0
    gray = array @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(gray)
    stats = {
        "luminance": float(gray.mean()),
        "luminance_sd": float(gray.std()),
        "edge_strength": float(np.hypot(gx, gy).mean()),
        "saturation": float("nan"),
        "blue_share": float("nan"),
    }
    if colour:
        hsv = np.asarray(rgb.convert("HSV"), dtype=np.float64) / 255.0
        total = array.sum(axis=2)
        lit = total > 0
        stats["saturation"] = float(hsv[..., 1].mean())
        stats["blue_share"] = float((array[..., 2][lit] / total[lit]).mean())
    return stats


def photo_statistics(ds: Dataset) -> tuple[list[dict], dict]:
    rows = []
    by_tool = metas_by_tool(ds.metas)
    for tool, metas in by_tool.items():
        lab = ds.labels[tool]
        position = {s.record_id: s.position for s in ds.timelines[tool] if s.recorded}
        smoothed = isotonic([m.flank_wear for m in metas])
        for meta, wear in zip(metas, smoothed, strict=True):
            row = {
                "tool": tool,
                "record_id": meta.record_id,
                "wear_smoothed_um": float(wear),
                "rul_pass_end": lab.ttf_position - position[meta.record_id]
                if position[meta.record_id] <= lab.ttf_position
                else None,
            }
            for kind in PHOTO_KINDS:
                for key, value in _photo_stats(
                    photo_path(paths.raw_dir(), meta.record_id, kind), kind
                ).items():
                    row[f"{kind}_{key}"] = value
            rows.append(row)
    keys = [
        k
        for k in rows[0]
        if "_" in k and k.split("_")[0] in PHOTO_KINDS and np.isfinite(rows[0][k])
    ]
    for tool in ds.tools:  # relative to the tool's first photo
        first = next(r for r in rows if r["tool"] == tool)
        for r in rows:
            if r["tool"] == tool:
                for key in keys:
                    r[f"rel_{key}"] = r[key] - first[key]
    pre = [r for r in rows if r["rul_pass_end"] is not None]
    summary = {
        key: {
            "rho_wear_pooled_absolute": round(
                rho([r[key] for r in rows], [r["wear_smoothed_um"] for r in rows]), 3
            ),
            "rho_wear_pooled_relative": round(
                rho([r[f"rel_{key}"] for r in rows], [r["wear_smoothed_um"] for r in rows]), 3
            ),
            "rho_rul_pooled_relative": round(
                rho([r[f"rel_{key}"] for r in pre], [r["rul_pass_end"] for r in pre]), 3
            ),
        }
        for key in keys
    }
    return rows, summary


STEP_STATS = (
    "work_luminance",
    "work_luminance_sd",
    "work_edge_strength",
    "chip_luminance",
    "chip_edge_strength",
    "chip_saturation",
    "chip_blue_share",
    "tool_luminance",
    "tool_edge_strength",
    "tool_saturation",
    "tool_blue_share",
    "rms_F",
    "rms_Fx",
    "rms_Fy",
    "rms_Fz_rel",
)


# --------------------------------------------------------------------------- figures


def fig_stage_step(ds: Dataset, photo_rows: list[dict], key: str, out: Path) -> None:
    by_tool = metas_by_tool(ds.metas)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.2), sharey=True)
    for ax, (tool, metas) in zip(axes.flat, by_tool.items(), strict=True):
        rows = [r for r in photo_rows if r["tool"] == tool]
        y = [r[key] for r in rows]
        ax.plot(range(1, len(y) + 1), y, color=BLUE, lw=LINE * 0.6, alpha=0.6)
        ax.scatter(range(1, len(y) + 1), y, s=MARK, color=BLUE, lw=0)
        first2 = next(i for i, m in enumerate(metas) if m.tool_label >= 2) + 1
        first3 = next(i for i, m in enumerate(metas) if m.tool_label >= 3) + 1
        for at, label, height in ((first2, "stage 2", 0.03), (first3, "stage 3", 0.13)):
            ax.axvline(at - 0.5, color=ORANGE, lw=LINE)
            ax.text(at, height, label, color=INK2, fontsize=7.5, transform=ax.get_xaxis_transform())
        ax.set_title(tool, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("workpiece photo mean luminance")
    fig.supxlabel(
        "recorded pass; orange: first pass of tool_label stage 2 and stage 3",
        fontsize=9,
        color=INK2,
    )
    fig.subplots_adjust(hspace=0.42)
    save(fig, out)


def fig_wear_curves(ds: Dataset, out: Path) -> None:
    by_tool = metas_by_tool(ds.metas)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharey=True)
    for ax, (tool, metas) in zip(axes.flat, by_tool.items(), strict=True):
        position = {s.record_id: s.position for s in ds.timelines[tool] if s.recorded}
        x = [position[m.record_id] for m in metas]
        wear = [m.flank_wear for m in metas]
        for slot in ds.timelines[tool]:
            if not slot.recorded:
                ax.axvspan(slot.position - 0.5, slot.position + 0.5, color=GRID, lw=0)
        gap = [s.position for s in ds.timelines[tool] if not s.recorded]
        if gap:
            ax.text(
                np.mean(gap), 330, "unrecorded", ha="center", va="top", color=INK2, fontsize=7.5
            )
        ax.scatter(x, wear, s=MARK, color=MUTED, lw=0, label="measured", zorder=2)
        ax.plot(x, isotonic(wear), color=BLUE, lw=LINE, label="isotonic smoothed", zorder=3)
        ax.axhline(110, color=MUTED, lw=0.7, ls=(0, (4, 3)), label="110 µm threshold", zorder=1)
        ttf = ds.labels[tool].ttf_position
        ax.axvline(ttf, color=ORANGE, lw=LINE, label="TTF (confirmed rule)", zorder=4)
        ax.text(ttf + 0.8, 18, f"TTF {ttf}", color=INK2, fontsize=7.5)
        ax.set_title(tool, loc="left")
        ax.set_ylim(0, 350)
    for ax in axes[1]:
        ax.set_xlabel("pass (gap-aware)")
    for ax in axes[:, 0]:
        ax.set_ylabel("flank wear (µm)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.subplots_adjust(hspace=0.42)
    save(fig, out)


def fig_trim(
    summary: dict, flagged: list[dict], ds: Dataset, out_hist: Path, out_review: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, key, title, limit in (
        (axes[0], "_start_diff", "Start: causal − OmniFuser", 400),
        (axes[1], "_end_diff", "End: causal − OmniFuser", 1200),
    ):
        values = np.clip(np.asarray(summary[key]), -limit, limit)
        ax.hist(
            values, bins=40, range=(-limit, limit), color=BLUE, edgecolor=SURFACE, linewidth=0.8
        )
        ax.set_title(title, loc="left")
        ax.set_xlabel("difference (samples, 1 ms each)")
    axes[0].set_ylabel("passes")
    save(fig, out_hist)

    review = [
        r for r in flagged if abs(r["causal_start"] - r["omnifuser_start"]) > 200 or r["in_eda_md"]
    ]
    review = sorted(review, key=lambda r: (not r["in_eda_md"], r["record_id"]))[:7]
    fig, axes = plt.subplots(1, len(review), figsize=(2.6 * len(review), 2.8), sharey=False)
    for ax, r in zip(np.atleast_1d(axes), review, strict=True):
        env = envelope(ds.store.force(r["record_id"]))[:15000]
        ax.plot(np.arange(len(env)) / 1000, env, color=MUTED, lw=0.6)
        ax.axvline(r["causal_start"] / 1000, color=BLUE, lw=LINE, label="causal start")
        ax.axvline(
            r["omnifuser_start"] / 1000,
            color=ORANGE,
            lw=LINE,
            ls=(0, (3, 2)),
            label="OmniFuser start",
        )
        ax.set_title(
            r["record_id"] + (" (EDA.md)" if r["in_eda_md"] else ""), loc="left", fontsize=8.5
        )
        ax.set_xlabel("seconds from recording start")
    np.atleast_1d(axes)[0].set_ylabel("|F| envelope (N)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    save(fig, out_review)


def fig_durations(ds: Dataset, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    rng = np.random.default_rng(0)
    for index, metas in enumerate(metas_by_tool(ds.metas).values()):
        d = np.array([ds.durations[m.record_id] for m in metas])
        ax.scatter(
            index + rng.uniform(-0.18, 0.18, len(d)), d, s=MARK, color=BLUE, lw=0, alpha=0.75
        )
        ax.plot([index - 0.3, index + 0.3], [np.median(d)] * 2, color=INK, lw=LINE)
    ax.set_xticks(range(len(ds.tools)), ds.tools)
    ax.set_ylabel("trimmed pass duration (s)")
    ax.set_title("Pass durations per tool (bar: median)", loc="left")
    save(fig, out)


def fig_force_vs_rul(rows: list[dict], ds: Dataset, key: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.4), sharey=True)
    for ax, tool in zip(axes.flat, ds.tools, strict=True):
        pre = [r for r in rows if r["tool"] == tool and r["rul_pass_end"] is not None]
        x = [r["rul_pass_end"] for r in pre]
        y = [100 * r[f"rel_{key}"] for r in pre]
        ax.plot(x, y, color=BLUE, lw=LINE * 0.6, alpha=0.6)
        ax.scatter(x, y, s=MARK, color=BLUE, lw=0)
        ax.axvspan(0, 15, color=GRID, lw=0, zorder=0)
        ax.invert_xaxis()
        ax.set_title(f"{tool}   ρ = {rho(y, x):+.2f}", loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("change from first run (%)")
    fig.supxlabel(
        "RUL at the end of the pass (passes; shaded: the k = 15 window)", fontsize=9, color=INK2
    )
    fig.suptitle(
        f"{FORCE_TITLES.get(key, key)} of the trimmed force, relative to each tool's first run",
        x=0.01,
        ha="left",
        y=1.0,
        fontsize=10,
    )
    fig.subplots_adjust(hspace=0.42)
    save(fig, out)


def fig_baselines(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    tools = list(dict.fromkeys(r["fold_test"] for r in rows))
    for ax, unit, label in (
        (axes[0], "pass", "MAE (passes)"),
        (axes[1], "sec", "MAE (s of cutting)"),
    ):
        for color, name, offset in ((BLUE, "B0", -0.2), (ORANGE, "B1", 0.0), (AQUA, "B2", 0.2)):
            values = [
                next(
                    r["mae"]
                    for r in rows
                    if r["fold_test"] == t and r["unit"] == unit and r["baseline"] == name
                )
                for t in tools
            ]
            ax.scatter(
                np.arange(len(tools)) + offset, values, s=MARK * 1.6, color=color, lw=0, label=name
            )
        ax.set_xticks(range(len(tools)), tools)
        ax.set_ylabel(label)
        ax.set_xlabel("test tool")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(
        "Usage baselines on each test tool's k-window", x=0.01, ha="left", y=1.06, fontsize=10
    )
    save(fig, out)


def fig_photos(rows: list[dict], out: Path) -> None:
    picks = (
        ("tool_edge_strength", "Tool photo: edge strength (edge band)"),
        ("chip_saturation", "Chip photo: mean saturation"),
        ("work_luminance_sd", "Workpiece photo: luminance sd"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    wear = [r["wear_smoothed_um"] for r in rows]
    for ax, (key, title) in zip(axes, picks, strict=True):
        values = [r[f"rel_{key}"] for r in rows]
        ax.scatter(wear, values, s=MARK, color=BLUE, lw=0, alpha=0.7)
        ax.set_title(f"{title}\nρ with wear = {rho(values, wear):+.2f}", loc="left")
        ax.set_xlabel("smoothed flank wear (µm)")
    axes[0].set_ylabel("change from the tool's first photo")
    save(fig, out)


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=paths.runs_dir() / "eda")
    parser.add_argument("--figures", type=Path, default=paths.REPO_ROOT / "docs" / "eda_rul")
    args = parser.parse_args(argv)
    print("resources:", check_resources(min_memory_mb=800).describe())
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    style()

    ds = load_dataset()
    ds_rec = load_dataset(DatasetOptions(axis="recorded"))
    folds = list(leave_one_tool_out_folds(ds.tools))
    summary: dict = {"options": vars(ds.options), "tools": ds.tools}

    print("gaps, duplicates, trims, durations")
    summary["gaps"] = audit.gap_wear_jumps(ds.metas)
    summary["duplicates"] = duplicate_analysis(ds)
    trim_summary, flagged = trim_analysis(ds)
    summary["trim"] = {k: v for k, v in trim_summary.items() if not k.startswith("_")}
    summary["trim_review"] = flagged
    summary["durations"] = duration_analysis(ds)

    print("TTF rules, windows, baselines")
    summary["ttf"] = ttf_analysis(ds, ds_rec)
    summary["windows_gap"] = window_analysis(ds)
    summary["windows_recorded"] = window_analysis(ds_rec)
    base_gap = score_usage_baselines(ds.labels, folds)
    base_rec = score_usage_baselines(ds_rec.labels, folds)
    summary["baselines"] = {"gap": base_gap["summary"], "recorded": base_rec["summary"]}
    write_csv(
        [{"axis": "gap", **r} for r in base_gap["rows"]]
        + [{"axis": "recorded", **r} for r in base_rec["rows"]],
        args.out / "baselines.csv",
    )

    print("force indicators")
    force_rows, summary["force_indicators"] = force_indicators(ds)
    write_csv(force_rows, args.out / "force_indicators.csv")

    print("photo statistics")
    photo_rows, summary["photo_statistics"] = photo_statistics(ds)
    write_csv(photo_rows, args.out / "photo_statistics.csv")

    print("stage-boundary step check")
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in [*photo_rows, *force_rows]:
        values[row["record_id"]].update({k: row[k] for k in STEP_STATS if k in row})
    summary["stage_steps"] = audit.stage_steps(ds.metas, values, STEP_STATS)

    for name in ("gaps", "durations", "windows_gap", "windows_recorded", "trim_review"):
        write_csv(summary[name], args.out / f"{name}.csv")

    print("figures")
    fig_wear_curves(ds, args.figures / "01_wear_curves.png")
    fig_trim(
        trim_summary,
        flagged,
        ds,
        args.figures / "02_trim_agreement.png",
        args.figures / "03_trim_review.png",
    )
    fig_durations(ds, args.figures / "04_durations.png")
    fig_force_vs_rul(force_rows, ds, "rms_F", args.figures / "05_force_rms_vs_rul.png")
    fig_baselines(base_gap["rows"], args.figures / "06_baselines.png")
    fig_photos(photo_rows, args.figures / "07_photo_statistics.png")
    fig_stage_step(ds, photo_rows, "work_luminance", args.figures / "08_workpiece_stage_step.png")

    summary["runtime_s"] = round(time.time() - started, 1)
    (args.out / "eda_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8"
    )
    print(f"wrote {args.out / 'eda_summary.json'} in {summary['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
