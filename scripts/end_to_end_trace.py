"""Trace one prediction point through the data pipeline, for docs/END_TO_END.md.

The point is the plan's worked example (PROJECT_PLAN.md §3.2): tool T5, pass 10, segment 5.
Every number of the document's walk-through comes from here. It reads the generic store
built through the Nonastreda adapter.

    D:\\RULModel\\.venv\\Scripts\\python.exe scripts/end_to_end_trace.py
"""

from __future__ import annotations

import json

import numpy as np

from rulfuse import paths
from rulfuse.adapters.nonastreda import load_data, to_dataset
from rulfuse.adapters.nonastreda.dataset import trimmed_bounds
from rulfuse.core.channels import DESCRIPTOR_NAMES, descriptors, expand, head_mean, model_channels
from rulfuse.core.normalization import relative_to_baseline
from rulfuse.core.store import SegmentStore, build_segment_store, image_visible
from rulfuse.features.handcrafted import segment_mean_rms

TOOL, POSITION, SEGMENT = "T5", 10, 5


def main() -> int:
    data = load_data()
    ds = to_dataset(data)
    store = SegmentStore.open(build_segment_store(ds))
    seg, rec, img = store.segments, store.recordings, store.images
    record = next(r.key for r in ds.unit(TOOL).recordings if r.attrs["position"] == POSITION)
    here_rows = (seg["unit"] == TOOL) & (seg["recording_key"] == record)
    row = int(np.flatnonzero(here_rows & (seg["segment"] == SEGMENT))[0])
    rrow = int(np.flatnonzero((rec["unit"] == TOOL) & (rec["key"] == record))[0])
    raw = data.store.force(str(seg["source"][row]))
    omni = trimmed_bounds(data.store, "omnifuser")[record]
    start, end = int(seg["start_sample"][row]), int(seg["end_sample"][row])
    trim_start = int(rec["start_sample"][rrow])
    head = head_mean(expand(raw[trim_start:], ds.channels), 1000.0)
    level = descriptors(expand(raw[start:end], ds.channels), 1000.0, head)
    names = [c.name for c in model_channels(ds.channels)]

    rows, features, stat_names = segment_mean_rms(store, ds.signals)
    mask = seg["unit"][rows] == TOOL
    relative = relative_to_baseline(features[mask], seg["in_baseline"][rows][mask])
    here = int(np.flatnonzero(rows[mask] == row)[0])

    tool_photos = store.image_rows(TOOL, "tool")
    visible = [
        str(img["key"][i])
        for i in tool_photos
        if image_visible(img["available_s"][i], seg["tau_s"][row])
    ]
    trace = {
        "store_signature": store.signature,
        "point": {"unit": TOOL, "pass": POSITION, "segment": SEGMENT, "record_id": record},
        "raw": {"samples": int(len(raw)), "seconds": len(raw) / 1000},
        "trim": {
            "causal": [trim_start, int(rec["end_sample"][rrow])],
            "omnifuser": list(omni),
            "duration_s": float(rec["duration_s"][rrow]),
        },
        "segments_in_recording": int(here_rows.sum()),
        "segment_samples": [start, end],
        "times": {
            "recording_start_s": float(rec["start_s"][rrow]),
            "elapsed_s": float(seg["elapsed_s"][row]),
            "tau_s": float(seg["tau_s"][row]),
        },
        "failure": store.units[TOOL],
        "targets": {k: float(seg[k][row]) for k in ("rul_s", "y_s")}
        | {k: bool(seg[k][row]) for k in ("window", "is_prediction_point")},
        "signal": {
            "signal_present": bool(seg["signal_present"][row]),
            "distinct_segment": int(seg["distinct_segment"][row]),
            "level_descriptors": {
                name: dict(zip(DESCRIPTOR_NAMES, np.round(level[c], 4).tolist(), strict=True))
                for c, name in enumerate(names)
            },
            "mean_rms": dict(
                zip(stat_names, np.round(features[mask][here], 3).tolist(), strict=True)
            ),
            "mean_rms_relative_to_baseline": dict(
                zip(stat_names, np.round(relative[here], 3).tolist(), strict=True)
            ),
        },
        "tool_photos_visible": {"count": len(visible), "last": visible[-1] if visible else None},
        "recording": {
            "hi_um": round(float(rec["hi"][rrow]), 2),
            "hi_causal_um": round(float(rec["hi_causal"][rrow]), 2),
            "stage": int(rec["stage"][rrow]),
            "image_label": str(rec["attr_image_label"][rrow]),
            "flank_wear_um": float(rec["attr_flank_wear_um"][rrow]),
        },
    }
    out = paths.runs_dir() / "stage3" / "trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace, indent=1, default=str), encoding="utf-8")
    print(json.dumps(trace, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
