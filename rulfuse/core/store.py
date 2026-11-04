"""The segment store (PROJECT_PLAN.md §5.8), built from any adapter's :class:`Dataset`.

``interim/segments/<dataset>/<signature>/`` holds tables; the signal samples stay in the
dataset's signal source, which every segment row points into.

* ``recordings.parquet``: one row per recording: unit, index, key, times, signal reference,
  HI (target and causal), stage, group, baseline flag, failure flag, and the adapter's
  ``attrs`` as ``attr_<name>`` columns (never inputs);
* ``segments.parquet``: one row per segment of every recording, including those after the
  failure (history tokens and HI targets): sample bounds, the distinct-segment id for
  encoding, the input times (τ and the seconds since the recording started), and the RUL
  targets of prediction points;
* ``images.parquet``: one row per image: unit, type, key, path, availability time;
* ``units.json``: per unit, the failure time, its recording and the baseline recordings;
* ``meta.json``: the signature, its payload, settings, channels, image specs and counts.

The signature hashes the adapter's provenance (source hashes and options), the settings,
the channels and the store version, so a store is never reused for other inputs.

Image rule (D9): an image is readable by a prediction at τ only if ``available_s < τ``
(``image_visible``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rulfuse import __version__, paths
from rulfuse.core.schema import (
    TIME_TOLERANCE_S,
    ChannelSpec,
    Dataset,
    ImageSpec,
    Settings,
    validate,
)
from rulfuse.core.segments import segment_ends, segment_sample_bounds
from rulfuse.core.targets import prediction_mask, rul_targets

STORE_VERSION = 2


def image_visible(available_s: np.ndarray | float, tau_s: np.ndarray | float):
    """True where an image available at ``available_s`` may be read by a prediction at τ."""
    return np.asarray(available_s) < np.asarray(tau_s) - TIME_TOLERANCE_S


def stable_hash(obj: object, length: int = 12) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def signature_payload(ds: Dataset) -> dict:
    return {
        "store_version": STORE_VERSION,
        "dataset": ds.name,
        "provenance": dict(ds.provenance),
        "settings": asdict(ds.settings),
        "channels": [asdict(c) for c in ds.channels],
        "image_specs": {kind: asdict(spec) for kind, spec in sorted(ds.image_specs.items())},
    }


def default_store_dir(ds: Dataset, sig: str) -> Path:
    return paths.interim_dir() / "segments" / ds.name / sig


def _attr_columns(rows: list[dict], records: list) -> None:
    keys = list(dict.fromkeys(key for rec in records for key in rec.attrs))
    for row, rec in zip(rows, records, strict=True):
        for key in keys:
            row[f"attr_{key}"] = rec.attrs.get(key)


def _tables(ds: Dataset) -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    s = ds.settings
    recording_rows: list[dict] = []
    segment_rows: list[dict] = []
    image_rows: list[dict] = []
    units: dict[str, dict] = {}
    distinct: dict[tuple[str, int, int, int], int] = {}
    for unit in ds.units:
        baseline = {rec.key for rec in unit.recordings[: s.baseline_recordings]}
        failure_key = ""
        for index, rec in enumerate(unit.recordings, start=1):
            is_failure = unit.failure_s is not None and (
                rec.start_s < unit.failure_s <= rec.end_s + TIME_TOLERANCE_S
            )
            failure_key = rec.key if is_failure and not failure_key else failure_key
            ref = rec.signal
            recording_rows.append(
                {
                    "unit": unit.name,
                    "index": index,
                    "key": rec.key,
                    "start_s": rec.start_s,
                    "end_s": rec.end_s,
                    "duration_s": rec.duration_s,
                    "signal_present": ref is not None,
                    "source": ref.source if ref else "",
                    "start_sample": ref.start if ref else -1,
                    "end_sample": ref.end if ref else -1,
                    "hi": rec.hi,
                    "hi_causal": rec.hi_causal,
                    "stage": -1 if rec.stage is None else int(rec.stage),
                    "group": rec.group,
                    "in_baseline": rec.key in baseline,
                    "is_failure_recording": is_failure,
                }
            )
            ends = segment_ends(rec.duration_s, s.segment_s, s.min_remainder_s)
            if ref is not None:
                bounds = [
                    (ref.start + a, ref.start + b)
                    for a, b in segment_sample_bounds(
                        ref.samples, s.sample_rate_hz, s.segment_s, s.min_remainder_s
                    )
                ]
                if len(bounds) != len(ends):
                    raise RuntimeError(f"{rec.key}: {len(bounds)} sample bounds, {len(ends)} ends")
            else:
                bounds = [(-1, -1)] * len(ends)
            tau = rec.start_s + ends
            point = prediction_mask(tau, unit.failure_s)
            targets = (
                rul_targets(tau[point], unit.failure_s, s.horizon_s)
                if unit.failure_s is not None and point.any()
                else None
            )
            target_index = np.cumsum(point) - 1
            for seg, (end, (start, stop)) in enumerate(zip(ends, bounds, strict=True), start=1):
                distinct_id = -1
                if ref is not None:
                    key = (ref.source, ref.start, start, stop)
                    distinct_id = distinct.setdefault(key, len(distinct))
                row_point = bool(point[seg - 1])
                t = int(target_index[seg - 1])
                segment_rows.append(
                    {
                        "unit": unit.name,
                        "recording": index,
                        "recording_key": rec.key,
                        "segment": seg,
                        "signal_present": ref is not None,
                        "source": ref.source if ref else "",
                        "recording_start_sample": ref.start if ref else -1,
                        "start_sample": start,
                        "end_sample": stop,
                        "distinct_segment": distinct_id,
                        "elapsed_s": float(end),
                        "tau_s": float(tau[seg - 1]),
                        "is_prediction_point": row_point,
                        "rul_s": float(targets.rul_s[t]) if row_point and targets else np.nan,
                        "y_s": float(targets.y_s[t]) if row_point and targets else np.nan,
                        "window": bool(targets.window[t]) if row_point and targets else False,
                        "in_baseline": rec.key in baseline and ref is not None,
                        "hi": rec.hi,
                    }
                )
        _attr_columns(recording_rows[-len(unit.recordings) :], list(unit.recordings))
        for image in unit.images:
            image_rows.append(
                {
                    "unit": unit.name,
                    "kind": image.kind,
                    "key": image.key,
                    "path": image.path,
                    "available_s": image.available_s,
                }
            )
        units[unit.name] = {
            "failure_s": unit.failure_s,
            "failure_recording": failure_key,
            "recordings": len(unit.recordings),
            "baseline_recordings": [rec.key for rec in unit.recordings[: s.baseline_recordings]],
            "attrs": dict(unit.attrs),
        }
    return recording_rows, segment_rows, image_rows, units


def _write(rows: list[dict], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    pq.write_table(pa.table({c: [row.get(c) for row in rows] for c in columns}), path)


def build_segment_store(ds: Dataset, root: str | Path | None = None, force: bool = False) -> Path:
    """Validate ``ds`` and write its store (atomically). An existing store with the same
    signature is reused unless ``force``."""
    validate(ds)
    payload = signature_payload(ds)
    sig = stable_hash(payload)
    out = Path(root) if root is not None else default_store_dir(ds, sig)
    if (out / "meta.json").is_file() and not force:
        existing = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        if existing.get("signature") == sig:
            return out
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    recordings, segments, images, units = _tables(ds)
    _write(recordings, tmp / "recordings.parquet")
    _write(segments, tmp / "segments.parquet")
    _write(images, tmp / "images.parquet")
    (tmp / "units.json").write_text(json.dumps(units, indent=1, default=str), encoding="utf-8")
    point = np.array([row["is_prediction_point"] for row in segments], dtype=bool)
    present = np.array([row["signal_present"] for row in segments], dtype=bool)
    counts = {
        "units": len(units),
        "recordings": len(recordings),
        "segments": len(segments),
        "prediction_points": int(point.sum()),
        "window_points": int(sum(row["window"] for row in segments)),
        "signal_missing_points": int((point & ~present).sum()),
        "distinct_segments": int(max((row["distinct_segment"] for row in segments), default=-1))
        + 1,
        "images": len(images),
        "images_by_kind": {
            kind: sum(row["kind"] == kind for row in images)
            for kind in sorted({row["kind"] for row in images})
        },
    }
    meta = {
        "signature": sig,
        "payload": payload,
        "counts": counts,
        "image_root": str(ds.image_root) if ds.image_root else None,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rulfuse_version": __version__,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp.rename(out)
    return out


def _read(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


@dataclass
class SegmentStore:
    root: Path
    meta: dict
    units: dict[str, dict]
    recordings: dict[str, np.ndarray]
    segments: dict[str, np.ndarray]
    images: dict[str, np.ndarray]

    @classmethod
    def open(cls, root: str | Path) -> SegmentStore:
        path = Path(root)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["payload"]["store_version"] != STORE_VERSION:
            raise ValueError(f"{path}: segment store version mismatch")
        units = json.loads((path / "units.json").read_text(encoding="utf-8"))
        return cls(
            path,
            meta,
            units,
            _read(path / "recordings.parquet"),
            _read(path / "segments.parquet"),
            _read(path / "images.parquet"),
        )

    @property
    def signature(self) -> str:
        return self.meta["signature"]

    @property
    def dataset(self) -> str:
        return self.meta["payload"]["dataset"]

    @property
    def settings(self) -> Settings:
        return Settings(**self.meta["payload"]["settings"])

    @property
    def channels(self) -> tuple[ChannelSpec, ...]:
        return tuple(ChannelSpec(**c) for c in self.meta["payload"]["channels"])

    @property
    def image_specs(self) -> dict[str, ImageSpec]:
        specs = {}
        for kind, spec in self.meta["payload"]["image_specs"].items():
            crop = tuple(spec["crop"]) if spec.get("crop") is not None else None
            specs[kind] = ImageSpec(spec["height"], spec["width"], crop, spec["standardize"])  # type: ignore[arg-type]
        return specs

    @property
    def unit_names(self) -> list[str]:
        return list(self.units)

    def segment_rows(self, unit: str) -> np.ndarray:
        return np.flatnonzero(self.segments["unit"] == unit)

    def recording_rows(self, unit: str) -> np.ndarray:
        return np.flatnonzero(self.recordings["unit"] == unit)

    def image_rows(self, unit: str | None = None, kind: str | None = None) -> np.ndarray:
        keep = np.ones(len(self.images["unit"]), dtype=bool)
        if unit is not None:
            keep &= self.images["unit"] == unit
        if kind is not None:
            keep &= self.images["kind"] == kind
        return np.flatnonzero(keep)

    def image_path(self, row: int) -> Path:
        path = Path(str(self.images["path"][row]))
        root = self.meta.get("image_root")
        return path if path.is_absolute() or not root else Path(root) / path
