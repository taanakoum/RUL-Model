"""Stage 2 validation (PROJECT_PLAN.md §15, P7 scientific check, and the photo preparation).

1. **Tool shift (P7).** Per fold, the test tool's mean position in the training tools'
   standardization, for the per-segment level and spread of the five force channels.
   Compared without and with the causal tool baseline (tokens relative to each tool's
   first 4 recorded passes). Acceptance: the relative version's median shift is below
   OmniFuser's 0.16-0.32 sd (EDA.md §8) and below the absolute version's.
2. **Workpiece standardization.** The Stage-1 stage-step probe (``rulfuse.audit``) on
   the raw brightness and on the standardized photos' edge strength. This previews the
   Gate-1 leakage probe.

    D:\\RULModel\\.venv\\Scripts\\python.exe scripts/validate_stage2.py
"""

from __future__ import annotations

import json
import time

import numpy as np

from rulfuse import audit, paths
from rulfuse.data.dataset import load_dataset
from rulfuse.data.folds import leave_one_tool_out_folds
from rulfuse.data.images import prepared_pixels
from rulfuse.data.nonastreda import photo_path
from rulfuse.data.normalization import relative_to_baseline, tool_shift
from rulfuse.data.segment_store import SegmentStore, build_segment_store
from rulfuse.features.physics import STAT_NAMES, segment_stats
from rulfuse.resources import check_resources


def edge_strength(pixels: np.ndarray) -> float:
    grey = pixels @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(grey)
    return float(np.hypot(gx, gy).mean())


def main() -> int:
    print("resources:", check_resources(min_memory_mb=800).describe())
    started = time.time()
    ds = load_dataset()
    store = SegmentStore.open(build_segment_store(ds))
    print(f"segment store {store.signature}")

    rows, absolute = segment_stats(store, ds.store)
    tools = store.segments["tool"][rows]
    in_base = store.segments["in_baseline"][rows]
    relative = np.empty_like(absolute)
    for tool in ds.tools:
        mask = tools == tool
        relative[mask] = relative_to_baseline(absolute[mask], in_base[mask])

    folds = list(leave_one_tool_out_folds(ds.tools))
    shift_rows = []
    for fold in folds:
        row: dict[str, float | str] = {"test": fold.test}
        for name, x in (("absolute", absolute), ("relative", relative)):
            z = tool_shift(x, tools, fold)
            row[f"{name}_median_abs"] = round(float(np.median(np.abs(z))), 3)
            row[f"{name}_max_abs"] = round(float(np.max(np.abs(z))), 3)
            row[f"{name}_mean_Fx"] = round(float(z[STAT_NAMES.index("mean_Fx")]), 3)
        shift_rows.append(row)
    shift = {
        "features": list(STAT_NAMES),
        "segments": len(rows),
        "per_fold": shift_rows,
        "median_over_folds_absolute": float(
            np.median([r["absolute_median_abs"] for r in shift_rows])
        ),
        "median_over_folds_relative": float(
            np.median([r["relative_median_abs"] for r in shift_rows])
        ),
        "folds_relative_below_absolute": int(
            sum(r["relative_median_abs"] < r["absolute_median_abs"] for r in shift_rows)
        ),
    }

    # Tool identifiability: nearest tool centroid, centroids from odd passes, scored on even
    # passes (chance 0.10). Relative tokens should make tools harder to tell apart.
    positions = store.segments["position"][rows]
    identify = {}
    for name, x in (("absolute", absolute), ("relative", relative)):
        z = (x - x.mean(axis=0)) / x.std(axis=0)
        fit, score = positions % 2 == 1, positions % 2 == 0
        names = sorted(set(tools), key=lambda t: int(t[1:]))
        centroids = np.stack([z[fit & (tools == t)].mean(axis=0) for t in names])
        distance = ((z[score][:, None, :] - centroids[None]) ** 2).sum(axis=2)
        predicted = np.asarray(names)[distance.argmin(axis=1)]
        identify[name] = round(float(np.mean(predicted == tools[score])), 3)
    shift["tool_identification_accuracy"] = identify

    values: dict[str, dict[str, float]] = {}
    for meta in ds.metas:
        path = photo_path(ds.raw, meta.record_id, "work")
        standardized = prepared_pixels(path, "work")
        grey = standardized @ np.array([0.299, 0.587, 0.114])
        values[meta.record_id] = {
            "raw_luminance": audit.mean_luminance(path),
            "standardized_luminance": float(grey.mean()),
            "standardized_luminance_sd": float(grey.std()),
            "standardized_edge_strength": edge_strength(standardized),
        }
    steps = audit.stage_steps(ds.metas, values, list(next(iter(values.values()))))

    result = {
        "segment_store": store.signature,
        "tool_shift": shift,
        "workpiece_stage_steps": steps,
        "runtime_s": round(time.time() - started, 1),
    }
    out = paths.runs_dir() / "stage2" / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"{'test':5s} {'abs median':>10s} {'rel median':>10s} {'abs Fx':>8s} {'rel Fx':>8s}")
    for r in shift_rows:
        print(
            f"{r['test']:5s} {r['absolute_median_abs']:10.3f} {r['relative_median_abs']:10.3f} "
            f"{r['absolute_mean_Fx']:8.3f} {r['relative_mean_Fx']:8.3f}"
        )
    print(
        f"median over folds: absolute {shift['median_over_folds_absolute']:.3f}, "
        f"relative {shift['median_over_folds_relative']:.3f}; relative lower in "
        f"{shift['folds_relative_below_absolute']} of 10 folds"
    )
    print(f"tool identification accuracy (chance 0.10): {identify}")
    for name, step in steps.items():
        print(
            f"workpiece {name:28s} 1to2 largest {step['largest_at_1to2']:2d}/10 "
            f"share {step['share_1to2']}  | 2to3 {step['largest_at_2to3']:2d}/10 "
            f"share {step['share_2to3']}"
        )
    print(f"wrote {out} in {result['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
