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
from rulfuse.adapters.nonastreda import load_data, to_dataset
from rulfuse.adapters.nonastreda.audit import stage_steps_by_record
from rulfuse.adapters.nonastreda.images import PHOTO_SPECS
from rulfuse.adapters.nonastreda.load import photo_path
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.images import prepared_pixels
from rulfuse.core.normalization import relative_to_baseline, unit_shift
from rulfuse.core.store import SegmentStore, build_segment_store
from rulfuse.features.handcrafted import segment_mean_rms
from rulfuse.resources import check_resources


def edge_strength(pixels: np.ndarray) -> float:
    grey = pixels @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(grey)
    return float(np.hypot(gx, gy).mean())


def main() -> int:
    print("resources:", check_resources(min_memory_mb=800).describe())
    started = time.time()
    ds = load_data()
    core = to_dataset(ds)
    store = SegmentStore.open(build_segment_store(core))
    print(f"segment store {store.signature}")

    rows, absolute, stat_names = segment_mean_rms(store, core.signals)
    tools = store.segments["unit"][rows]
    in_base = store.segments["in_baseline"][rows]
    relative = np.empty_like(absolute)
    for tool in ds.tools:
        mask = tools == tool
        relative[mask] = relative_to_baseline(absolute[mask], in_base[mask])

    folds = leave_one_unit_out(ds.tools)
    shift_rows = []
    for fold in folds:
        row: dict[str, float | str] = {"test": fold.test}
        for name, x in (("absolute", absolute), ("relative", relative)):
            z = unit_shift(x, tools, fold)
            row[f"{name}_median_abs"] = round(float(np.median(np.abs(z))), 3)
            row[f"{name}_max_abs"] = round(float(np.max(np.abs(z))), 3)
            row[f"{name}_mean_Fx"] = round(float(z[stat_names.index("mean_Fx")]), 3)
        shift_rows.append(row)
    shift = {
        "features": list(stat_names),
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
    positions = store.segments["recording"][rows]
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
        standardized = prepared_pixels(path, PHOTO_SPECS["work"])
        grey = standardized @ np.array([0.299, 0.587, 0.114])
        values[meta.record_id] = {
            "raw_luminance": audit.mean_luminance(path),
            "standardized_luminance": float(grey.mean()),
            "standardized_luminance_sd": float(grey.std()),
            "standardized_edge_strength": edge_strength(standardized),
        }
    steps = stage_steps_by_record(ds.metas, values, list(next(iter(values.values()))))

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
            f"workpiece {name:28s} largest at a stage boundary "
            f"{step['largest_at_stage_boundary']:2d}/10 {step['largest_by_transition']} "
            f"share {step['median_share']}"
        )
    print(f"wrote {out} in {result['runtime_s']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
