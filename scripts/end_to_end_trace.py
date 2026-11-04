"""Trace one prediction point through the data pipeline, for docs/END_TO_END.md.

The point is the plan's worked example (PROJECT_PLAN.md §3.2): tool T5, pass 10, segment 5.
Every number of the document's walk-through comes from here.

    D:\\RULModel\\.venv\\Scripts\\python.exe scripts/end_to_end_trace.py
"""

from __future__ import annotations

import json

import numpy as np

from rulfuse import paths
from rulfuse.data.channels import derived_channels, fz_offset
from rulfuse.data.dataset import load_dataset, trimmed_bounds
from rulfuse.data.normalization import relative_to_baseline
from rulfuse.data.segment_store import SegmentStore, build_segment_store, photo_visible
from rulfuse.features.physics import STAT_NAMES, channel_stats, segment_stats

TOOL, POSITION, SEGMENT = "T5", 10, 5


def main() -> int:
    ds = load_dataset()
    store = SegmentStore.open(build_segment_store(ds))
    seg, passes = store.segments, store.passes
    row = int(
        np.flatnonzero(
            (seg["tool"] == TOOL) & (seg["position"] == POSITION) & (seg["segment"] == SEGMENT)
        )[0]
    )
    prow = int(np.flatnonzero((passes["tool"] == TOOL) & (passes["position"] == POSITION))[0])
    record = str(seg["record_id"][row])
    rep = str(seg["representative"][row]) or record
    raw = ds.store.force(rep)
    omni = trimmed_bounds(ds.store, "omnifuser")[record]
    start, end = int(seg["start_sample"][row]), int(seg["end_sample"][row])
    trim_start = int(passes["trim_start"][prow])
    offset = fz_offset(raw[trim_start:])
    stats = channel_stats(derived_channels(raw[start:end], offset))

    rows, features = segment_stats(store, ds.store)
    tool_rows = rows[seg["tool"][rows] == TOOL]
    mask = seg["tool"][rows] == TOOL
    relative = relative_to_baseline(features[mask], seg["in_baseline"][rows][mask])
    here = int(np.flatnonzero(tool_rows == row)[0])

    tool_passes = np.flatnonzero(passes["tool"] == TOOL)
    visible = [
        int(passes["position"][i])
        for i in tool_passes
        if passes["recorded"][i] and photo_visible(POSITION, int(passes["photo_position"][i]))
    ]
    trace = {
        "store_signature": store.signature,
        "point": {"tool": TOOL, "position": POSITION, "segment": SEGMENT, "record_id": record},
        "raw": {"samples": int(len(raw)), "seconds": len(raw) / 1000},
        "trim": {
            "causal": [trim_start, int(passes["trim_end"][prow])],
            "omnifuser": list(omni),
            "duration_s": float(passes["duration_s"][prow]),
        },
        "segments_in_pass": int(np.sum((seg["tool"] == TOOL) & (seg["position"] == POSITION))),
        "segment_samples": [start, end],
        "times": {
            "elapsed_in_pass_s": float(seg["elapsed_in_pass_s"][row]),
            "completed_passes": float(seg["completed_passes"][row]),
            "tau_s": float(seg["tau_s"][row]),
            "t_label": float(seg["t_label"][row]),
            "mean_past_duration_s": float(seg["mean_past_duration_s"][row]),
        },
        "ttf": store.tools[TOOL],
        "targets": {k: float(seg[k][row]) for k in ("rul_pass", "rul_sec", "y_pass", "y_sec")}
        | {k: bool(seg[k][row]) for k in ("window_pass", "window_sec", "is_prediction_point")},
        "force": {
            "force_present": bool(seg["force_present"][row]),
            "distinct_segment": int(seg["distinct_segment"][row]),
            "fz_offset": round(offset, 3),
            "absolute": dict(zip(STAT_NAMES, np.round(stats, 3).tolist(), strict=True)),
            "relative_to_baseline": dict(
                zip(STAT_NAMES, np.round(relative[here], 3).tolist(), strict=True)
            ),
        },
        "photos_visible_from_passes": visible,
        "pass_labels": {
            "image_label": str(passes["image_label"][prow]),
            "flank_wear_um": float(passes["flank_wear_um"][prow]),
            "wear_smoothed_um": round(float(passes["wear_smoothed_um"][prow]), 2),
            "wear_causal_um": round(float(passes["wear_causal_um"][prow]), 2),
        },
    }
    out = paths.runs_dir() / "stage2" / "trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=1), encoding="utf-8")
    print(json.dumps(trace, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
