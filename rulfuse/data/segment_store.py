"""The segment store (PROJECT_PLAN.md §5.8, Stage 2).

``interim/segments/<signature>/`` holds the timeline and every 10 s segment as tables. The
force samples stay in the force store (``interim/forces``), which every segment row points
into, so nothing is copied.

* ``passes.parquet``: one row per pass on the axis, including unrecorded passes: identity,
  trim bounds, duration, cumulative cutting time, force attachment, photo availability,
  labels (targets only) and smoothed wear;
* ``segments.parquet``: one row per segment of every recorded pass, including those after
  the TTF (history tokens and wear targets): sample bounds in the force store, the
  distinct-recording id for encoding, input times, label time and the RUL targets of
  prediction points (before or at the TTF);
* ``tools.json``: per tool, the TTF and the first recorded passes (the causal baseline);
* ``meta.json``: the signature, the inputs it was computed from, and the counts.

Signature: a hash of the force store's source sha256, the label files' sha256, every
``DatasetOptions`` field and the segment settings, so a store is never reused for other
inputs or settings.

Photo rule (D9): the photo of pass *p* may be read by a prediction point of pass *q* only
if q > p (``photo_visible``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rulfuse import __version__, paths
from rulfuse.data.dataset import Dataset
from rulfuse.data.force_store import sha256_file
from rulfuse.data.labels import isotonic, isotonic_causal, prediction_points, rul_targets
from rulfuse.data.nonastreda import PHOTO_KINDS, SAMPLE_RATE_HZ, metas_by_tool
from rulfuse.data.segments import MIN_REMAINDER_SECONDS, SEGMENT_SECONDS, segment_sample_bounds

SEGMENT_STORE_VERSION = 1
BASELINE_PASSES = 4


def photo_visible(point_position: np.ndarray | int, photo_position: np.ndarray | int):
    """True where a point of pass ``point_position`` may read the photo of ``photo_position``."""
    return np.asarray(point_position) > np.asarray(photo_position)


def signature_payload(ds: Dataset) -> dict:
    return {
        "segment_store_version": SEGMENT_STORE_VERSION,
        "force_store": {
            "store_version": ds.store.meta["store_version"],
            "source_sha256": ds.store.meta["source_sha256"],
            "distinct_recordings": ds.store.meta["distinct_recordings"],
        },
        "labels_sha256": {
            name: sha256_file(ds.raw / name) for name in ("labels.csv", "labels_reg.csv")
        },
        "options": asdict(ds.options),
        "segment_s": SEGMENT_SECONDS,
        "min_remainder_s": MIN_REMAINDER_SECONDS,
        "baseline_passes": BASELINE_PASSES,
    }


def signature(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def default_store_dir(sig: str) -> Path:
    return paths.interim_dir() / "segments" / sig


def _pass_table(ds: Dataset) -> tuple[dict[str, list], dict[str, dict]]:
    by_tool = metas_by_tool(ds.metas)
    group_of = {member: index for index, group in enumerate(ds.store.groups) for member in group}
    rows: dict[str, list] = {
        key: []
        for key in (
            "tool",
            "position",
            "run",
            "block",
            "record_id",
            "recorded",
            "force_present",
            "force_record",
            "representative",
            "trim_start",
            "trim_end",
            "duration_s",
            "tau_start_s",
            "tau_end_s",
            "photo_position",
            "image_label",
            "tool_label",
            "flank_wear_um",
            "wear_smoothed_um",
            "wear_causal_um",
            "is_ttf_pass",
            "duplicate_group",
            *[f"photo_{kind}" for kind in PHOTO_KINDS],
        )
    }
    tools: dict[str, dict] = {}
    for tool, timeline in ds.timelines.items():
        labels = ds.labels[tool]
        recorded = [slot for slot in timeline if slot.recorded]
        meta_of = {meta.record_id: meta for meta in by_tool[tool]}
        wear = [meta_of[slot.record_id].flank_wear for slot in recorded]  # type: ignore[index]
        smoothed = dict(zip([s.record_id for s in recorded], isotonic(wear), strict=True))
        causal = dict(zip([s.record_id for s in recorded], isotonic_causal(wear), strict=True))
        for slot in timeline:
            rid = slot.record_id
            meta = meta_of.get(rid) if rid else None
            start, end = ds.bounds[rid] if rid else (-1, -1)
            rows["tool"].append(tool)
            rows["position"].append(slot.position)
            rows["run"].append(slot.run)
            rows["block"].append(slot.block)
            rows["record_id"].append(rid or "")
            rows["recorded"].append(slot.recorded)
            rows["force_present"].append(slot.force_present)
            rows["force_record"].append(slot.force_record or "")
            rows["representative"].append(
                ds.store.representative[slot.force_record] if slot.force_record else ""
            )
            rows["trim_start"].append(start)
            rows["trim_end"].append(end)
            rows["duration_s"].append(slot.duration_s)
            rows["tau_start_s"].append(slot.tau_start_s)
            rows["tau_end_s"].append(slot.tau_end_s)
            rows["photo_position"].append(slot.position if slot.recorded else -1)
            rows["image_label"].append(meta.image_label if meta else "")
            rows["tool_label"].append(meta.tool_label if meta else -1)
            rows["flank_wear_um"].append(meta.flank_wear if meta else float("nan"))
            rows["wear_smoothed_um"].append(
                smoothed.get(rid, float("nan")) if rid else float("nan")
            )
            rows["wear_causal_um"].append(causal.get(rid, float("nan")) if rid else float("nan"))
            rows["is_ttf_pass"].append(slot.position == labels.ttf_position)
            rows["duplicate_group"].append(group_of.get(rid, -1) if rid else -1)
            for kind, ext in PHOTO_KINDS.items():
                rows[f"photo_{kind}"].append(f"{kind}/{rid}{ext}" if rid else "")
        tools[tool] = {
            "ttf_position": labels.ttf_position,
            "ttf_record": labels.ttf_record,
            "tau_ttf_s": labels.tau_ttf,
            "wear_at_ttf_um": labels.wear_at_ttf_um,
            "last_position": timeline[-1].position,
            "recorded_passes": len(recorded),
            "unrecorded_passes": len(timeline) - len(recorded),
            "baseline_positions": [slot.position for slot in recorded[:BASELINE_PASSES]],
        }
    return rows, tools


def _segment_table(ds: Dataset) -> dict[str, list]:
    columns: dict[str, list] = {
        key: []
        for key in (
            "tool",
            "position",
            "segment",
            "record_id",
            "force_present",
            "representative",
            "start_sample",
            "end_sample",
            "distinct_segment",
            "elapsed_in_pass_s",
            "tau_s",
            "completed_passes",
            "mean_past_duration_s",
            "t_label",
            "duration_s",
            "is_prediction_point",
            "rul_pass",
            "rul_sec",
            "y_pass",
            "y_sec",
            "window_pass",
            "window_sec",
            "in_baseline",
        )
    }
    distinct: dict[tuple[str, int, int], int] = {}
    for tool, timeline in ds.timelines.items():
        labels = ds.labels[tool]
        points = prediction_points(timeline, last_position=timeline[-1].position)
        is_point = points.position <= labels.ttf_position
        targets = rul_targets(
            _subset(points, is_point),
            labels.ttf_position,
            labels.tau_ttf,
            ds.options.k,
            ds.options.k_sec,
        )
        recorded = [slot for slot in timeline if slot.recorded]
        baseline = {slot.position for slot in recorded[:BASELINE_PASSES]}
        bounds: list[tuple[int, int]] = []
        for slot in recorded:
            start, end = ds.bounds[slot.record_id]  # type: ignore[index]
            bounds.extend((start + a, start + b) for a, b in segment_sample_bounds(end - start))
        if len(bounds) != len(points):
            raise RuntimeError(f"{tool}: {len(bounds)} sample bounds but {len(points)} points")
        slot_of = {slot.position: slot for slot in recorded}
        target_index = np.cumsum(is_point) - 1
        for row in range(len(points)):
            slot = slot_of[int(points.position[row])]
            start, end = bounds[row]
            rep = ds.store.representative[slot.force_record] if slot.force_record else ""
            distinct_id = -1
            if slot.force_present:
                distinct_id = distinct.setdefault((rep, start, end), len(distinct))
            point = bool(is_point[row])
            t = int(target_index[row])
            columns["tool"].append(tool)
            columns["position"].append(int(points.position[row]))
            columns["segment"].append(int(points.segment[row]))
            columns["record_id"].append(points.record_id[row])
            columns["force_present"].append(bool(points.force_present[row]))
            columns["representative"].append(rep)
            columns["start_sample"].append(start)
            columns["end_sample"].append(end)
            columns["distinct_segment"].append(distinct_id)
            columns["elapsed_in_pass_s"].append(float(points.elapsed_in_pass[row]))
            columns["tau_s"].append(float(points.tau[row]))
            columns["completed_passes"].append(float(points.completed_passes[row]))
            columns["mean_past_duration_s"].append(float(points.mean_past_duration[row]))
            columns["t_label"].append(float(points.t_label[row]))
            columns["duration_s"].append(float(points.duration[row]))
            columns["is_prediction_point"].append(point)
            for name in ("rul_pass", "rul_sec", "y_pass", "y_sec"):
                columns[name].append(float(getattr(targets, name)[t]) if point else float("nan"))
            columns["window_pass"].append(bool(targets.window_pass[t]) if point else False)
            columns["window_sec"].append(bool(targets.window_sec[t]) if point else False)
            columns["in_baseline"].append(slot.position in baseline and slot.force_present)
    return columns


def _subset(points, mask: np.ndarray):
    return replace(
        points,
        **{
            name: getattr(points, name)[mask]
            for name in (
                "position",
                "segment",
                "elapsed_in_pass",
                "duration",
                "t_label",
                "tau",
                "completed_passes",
                "mean_past_duration",
                "force_present",
            )
        },
        record_id=tuple(rid for rid, keep in zip(points.record_id, mask, strict=True) if keep),
    )


def build_segment_store(ds: Dataset, root: str | Path | None = None, force: bool = False) -> Path:
    """Write the store for ``ds`` (atomically). An existing complete store with the same
    signature is reused unless ``force``."""
    payload = signature_payload(ds)
    sig = signature(payload)
    out = Path(root) if root is not None else default_store_dir(sig)
    if (out / "meta.json").is_file() and not force:
        existing = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        if existing.get("signature") == sig:
            return out
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    pass_rows, tools = _pass_table(ds)
    segment_rows = _segment_table(ds)
    pq.write_table(pa.table(pass_rows), tmp / "passes.parquet")
    pq.write_table(pa.table(segment_rows), tmp / "segments.parquet")
    (tmp / "tools.json").write_text(json.dumps(tools, indent=1), encoding="utf-8")

    point = np.asarray(segment_rows["is_prediction_point"], dtype=bool)
    present = np.asarray(segment_rows["force_present"], dtype=bool)
    counts = {
        "tools": len(tools),
        "passes": len(pass_rows["tool"]),
        "recorded_passes": int(np.sum(pass_rows["recorded"])),
        "unrecorded_passes": int(len(pass_rows["tool"]) - np.sum(pass_rows["recorded"])),
        "segments": len(segment_rows["tool"]),
        "prediction_points": int(point.sum()),
        "window_points_pass": int(np.sum(segment_rows["window_pass"])),
        "window_points_sec": int(np.sum(segment_rows["window_sec"])),
        "force_missing_points": int((point & ~present).sum()),
        "distinct_segments": int(max(segment_rows["distinct_segment"]) + 1),
    }
    meta = {
        "signature": sig,
        "payload": payload,
        "counts": counts,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "force_store": str(ds.store.root),
        "raw": str(ds.raw),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rulfuse_version": __version__,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def _read(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


@dataclass
class SegmentStore:
    root: Path
    meta: dict
    tools: dict[str, dict]
    passes: dict[str, np.ndarray]
    segments: dict[str, np.ndarray]

    @classmethod
    def open(cls, root: str | Path) -> SegmentStore:
        path = Path(root)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["payload"]["segment_store_version"] != SEGMENT_STORE_VERSION:
            raise ValueError(f"{path}: segment store version mismatch")
        tools = json.loads((path / "tools.json").read_text(encoding="utf-8"))
        return cls(
            path, meta, tools, _read(path / "passes.parquet"), _read(path / "segments.parquet")
        )

    @property
    def signature(self) -> str:
        return self.meta["signature"]

    def segment_rows(self, tool: str) -> np.ndarray:
        return np.flatnonzero(self.segments["tool"] == tool)

    def pass_rows(self, tool: str) -> np.ndarray:
        return np.flatnonzero(self.passes["tool"] == tool)
