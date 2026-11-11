"""The Stage-3 feature check (PROJECT_PLAN.md §6.2; parts P8, P9). No ranking of encoders.

1. **Scale check** (needs the encoder): is a window's embedding unchanged when the window
   is doubled? Measured against the spread between different windows.
2. **Information check (go/no-go)**: a ridge probe from recording-level pooled features to
   the HI, α by inner leave-one-unit-out, scored on each fold's validation unit against the
   training-mean predictor; with a shuffled-target control. Signal features (encoder +
   descriptors) and each image type must beat the mean. Descriptors alone and the encoder
   alone are reported too.
3. **HI readouts**: cross-fitted (leave-one-unit-out) readouts per image type and for the
   signals; MAE per unit.
4. **Leakage probe**: :func:`rulfuse.audit.stage_steps` on each image type's first 3
   principal components (PCA fitted without the unit it projects) and on its HI readout.
   A type leaks if a stage boundary is its largest group-to-group step in more than
   ``LEAK_MAX_UNITS`` units. The signals and the HI itself are reported as references.

Everything here reads only the caches and the store, so it runs anywhere; the scale check
and the determinism check run where the encoder runs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

from rulfuse.audit import LEAK_MAX_UNITS, stage_steps
from rulfuse.core.channels import DESCRIPTOR_NAMES, model_channels
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.normalization import PCA, Scaler
from rulfuse.core.schema import SignalSource
from rulfuse.core.store import SegmentStore
from rulfuse.features.cache import (
    ImageFeatures,
    SignalFeatures,
    _segment_inputs,
    distinct_rows,
    windows,
)
from rulfuse.features.probe import cross_fitted_readout, probe_folds

SHUFFLE_SEED = 0


# ----------------------------------------------------------------------------- pooling


def recording_signal_features(
    store: SegmentStore, features: SignalFeatures
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Recording rows with a signal, and their segment-averaged features:
    ``encoder`` (channels × 2·dim), ``descriptors`` (channels × 7)."""
    rec, seg = store.recordings, store.segments
    rows = np.flatnonzero(rec["signal_present"].astype(bool))
    key_to_row = {
        (str(u), int(i)): r for r, (u, i) in enumerate(zip(rec["unit"], rec["index"], strict=True))
    }
    sums: dict[int, list[np.ndarray]] = {}
    for s in np.flatnonzero(seg["distinct_segment"] >= 0):
        r = key_to_row[(str(seg["unit"][s]), int(seg["recording"][s]))]
        sums.setdefault(r, []).append(np.asarray(seg["distinct_segment"][s]))
    enc = np.empty((len(rows), int(np.prod(features.embeddings.shape[1:]))), dtype=np.float32)
    desc = np.empty((len(rows), int(np.prod(features.descriptors.shape[1:]))), dtype=np.float32)
    for i, r in enumerate(rows):
        ids = np.asarray(sums[int(r)], dtype=np.int64)
        enc[i] = np.asarray(features.embeddings[ids], dtype=np.float32).mean(axis=0).reshape(-1)
        desc[i] = np.asarray(features.descriptors[ids], dtype=np.float32).mean(axis=0).reshape(-1)
    return rows, {"encoder": enc, "descriptors": desc}


def image_vectors(
    store: SegmentStore, features: ImageFeatures, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """(image rows of ``kind``, CLS ⊕ mean of the grid tokens) per image."""
    rows = store.image_rows(kind=kind)
    tokens = np.asarray(features.tokens[rows], dtype=np.float32)
    return rows, np.concatenate([tokens[:, 0], tokens[:, 1:].mean(axis=1)], axis=1)


def _recording_of_images(store: SegmentStore, image_rows: np.ndarray) -> np.ndarray:
    rec = store.recordings
    index = {
        (str(u), str(k)): r for r, (u, k) in enumerate(zip(rec["unit"], rec["key"], strict=True))
    }
    return np.array(
        [
            index.get((str(store.images["unit"][r]), str(store.images["key"][r])), -1)
            for r in image_rows
        ]
    )


# ----------------------------------------------------------------------------- pieces


def loo_pcs(x: np.ndarray, units: np.ndarray, k: int = 3) -> np.ndarray:
    """First k principal components of every row, from a PCA fitted without its unit."""
    out = np.zeros((len(x), k))
    for unit in dict.fromkeys(units.tolist()):
        own = units == unit
        scaler = Scaler.fit(x[~own])
        pca = PCA.fit(scaler.transform(x[~own]), k)
        out[own] = pca.transform(scaler.transform(x[own]))
    return out


def _leakage(store: SegmentStore, rec_rows: np.ndarray, x: np.ndarray, readout: np.ndarray) -> dict:
    rec = store.recordings
    units = rec["unit"][rec_rows].astype(str)
    pcs = loo_pcs(x, units)
    values = {f"pc{i + 1}": pcs[:, i] for i in range(pcs.shape[1])}
    values["hi_readout"] = readout
    result = stage_steps(
        units, rec["group"][rec_rows].astype(str), rec["stage"][rec_rows].astype(int), values
    )
    worst = max(v["largest_at_stage_boundary"] for v in result.values())
    return {
        "statistics": result,
        "worst_units_at_stage_boundary": worst,
        "leaks": worst > LEAK_MAX_UNITS,
    }


def _readout_summary(y: np.ndarray, readout: np.ndarray, units: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(readout)
    per_unit = {
        str(u): round(float(np.mean(np.abs(readout[ok & (units == u)] - y[ok & (units == u)]))), 2)
        for u in dict.fromkeys(units.tolist())
        if (ok & (units == u)).any()
    }
    return {"mae": round(float(np.mean(np.abs(readout[ok] - y[ok]))), 2), "mae_per_unit": per_unit}


def scale_check(
    store: SegmentStore, signals: SignalSource, encoder, n: int = 200, seed: int = 0
) -> dict:
    """Relative embedding change when a window is doubled, against the relative distance
    between embeddings of different windows (median over ``n`` windows)."""
    rng = np.random.default_rng(seed)
    first = distinct_rows(store)
    picks = rng.choice(len(first), size=min(n, len(first)), replace=False)
    channels = model_channels(store.channels)
    x = []
    cache: dict = {}
    for d in picks:
        segment, _ = _segment_inputs(store, signals, int(first[d]), cache)
        w = windows(segment, encoder.length)  # (C, k, L)
        c = int(rng.integers(len(channels)))
        x.append(w[c, int(rng.integers(w.shape[1]))])
    x = np.stack(x)
    e1, e2 = encoder.encode(x), encoder.encode(2.0 * x)
    norm = np.linalg.norm(e1, axis=1) + 1e-12
    change = np.linalg.norm(e2 - e1, axis=1) / norm
    shuffled = e1[rng.permutation(len(e1))]
    spread = np.linalg.norm(shuffled - e1, axis=1) / norm
    cosine = (e1 * e2).sum(axis=1) / (norm * (np.linalg.norm(e2, axis=1) + 1e-12))
    ratio = float(np.median(change) / max(np.median(spread), 1e-12))
    return {
        "windows": len(x),
        "median_relative_change_2x": round(float(np.median(change)), 4),
        "median_relative_distance_between_windows": round(float(np.median(spread)), 4),
        "change_over_spread": round(ratio, 4),
        "median_cosine_2x": round(float(np.median(cosine)), 4),
        "scale_invariant": ratio < 0.1,
    }


def determinism_check(
    store: SegmentStore,
    signals: SignalSource,
    encoder,
    features: SignalFeatures,
    n: int = 16,
    seed: int = 1,
) -> dict:
    """Re-encode ``n`` cached segments one at a time (other batch sizes than the cache's).
    On a GPU, batch-dependent kernels (and TF32 convolutions) change the last digits, so the
    tolerance is relative: the largest difference over the largest cached magnitude."""
    rng = np.random.default_rng(seed)
    first = distinct_rows(store)
    picks = rng.choice(len(first), size=min(n, len(first)), replace=False)
    worst, scale, cache = 0.0, 0.0, {}
    for d in picks:
        segment, _ = _segment_inputs(store, signals, int(first[d]), cache)
        w = windows(segment, encoder.length)
        e = encoder.encode(w.reshape(-1, encoder.length)).reshape(w.shape[0], w.shape[1], -1)
        pooled = np.concatenate([e.mean(axis=1), e.std(axis=1)], axis=1)
        cached = np.asarray(features.embeddings[d], dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(pooled - cached))))
        scale = max(scale, float(np.max(np.abs(cached))))
    relative = worst / max(scale, 1e-12)
    return {
        "segments": int(len(picks)),
        "max_abs_difference": worst,
        "max_relative_difference": relative,
        "ok": relative < 1e-3,
    }


# ----------------------------------------------------------------------------- report


def feature_check(
    store: SegmentStore,
    signal_features: SignalFeatures,
    image_features: ImageFeatures | None,
    kinds: Sequence[str] | None = None,
    extra: dict | None = None,
) -> dict:
    rec = store.recordings
    hi = rec["hi"].astype(float)
    folds = leave_one_unit_out(store.unit_names)
    report: dict = {
        "store": store.signature,
        "dataset": store.dataset,
        "hi": store.settings.hi_name,
        "signal_features": str(signal_features.folder),
        "image_features": str(image_features.folder) if image_features else None,
        "leak_rule": (
            f"a stage boundary is the largest group step in more than {LEAK_MAX_UNITS} units"
        ),
        **(extra or {}),
    }
    rows, pooled = recording_signal_features(store, signal_features)
    units = rec["unit"][rows].astype(str)
    y = hi[rows]
    both = np.concatenate([pooled["encoder"], pooled["descriptors"]], axis=1)
    sets = {
        "encoder+descriptors": both,
        "encoder": pooled["encoder"],
        "descriptors": pooled["descriptors"],
    }
    signal = {name: probe_folds(x, y, units, folds) for name, x in sets.items()}
    signal["encoder+descriptors"]["shuffled_control"] = probe_folds(
        both, y, units, folds, shuffle_seed=SHUFFLE_SEED
    )
    readout = cross_fitted_readout(both, y, units, store.unit_names)
    channels = [c.name for c in model_channels(store.channels)]
    rms = pooled["descriptors"].reshape(len(rows), len(channels), len(DESCRIPTOR_NAMES))[:, :, 0]
    level = {
        f"spearman_log10_rms_{name}_vs_hi": round(
            float(np.asarray(spearmanr(rms[:, c], y, nan_policy="omit")[0])), 3
        )
        for c, name in enumerate(channels)
    }
    report["signals"] = {
        "recordings": int(len(rows)),
        "probe": signal,
        "readout": _readout_summary(y, readout, units),
        "level_vs_hi": level,
        "leakage_reference": _leakage(store, rows, both, readout),
        "go": bool(signal["encoder+descriptors"]["beats_mean"]),
    }
    # The HI itself, per recording: how often does real wear step most at a stage boundary?
    all_rows = np.flatnonzero(np.isfinite(hi))
    report["hi_reference"] = stage_steps(
        rec["unit"][all_rows].astype(str),
        rec["group"][all_rows].astype(str),
        rec["stage"][all_rows].astype(int),
        {"hi": hi[all_rows]},
    )
    images: dict = {}
    if image_features is not None:
        for kind in kinds or sorted(store.image_specs):
            img_rows, x = image_vectors(store, image_features, kind)
            rec_rows = _recording_of_images(store, img_rows)
            keep = rec_rows >= 0
            img_rows, x, rec_rows = img_rows[keep], x[keep], rec_rows[keep]
            u = rec["unit"][rec_rows].astype(str)
            target = hi[rec_rows]
            probe = probe_folds(x, target, u, folds)
            probe["shuffled_control"] = probe_folds(x, target, u, folds, shuffle_seed=SHUFFLE_SEED)
            out = cross_fitted_readout(x, target, u, store.unit_names)
            leak = _leakage(store, rec_rows, x, out)
            images[kind] = {
                "images": int(len(img_rows)),
                "probe": probe,
                "readout": _readout_summary(target, out, u),
                "leakage": leak,
                "informative": bool(probe["beats_mean"]),
                "kept": bool(probe["beats_mean"]) and not leak["leaks"],
            }
    report["images"] = images
    report["decisions"] = {
        "signals_go": report["signals"]["go"],
        "image_types_kept": [k for k, v in images.items() if v["kept"]],
        "image_types_excluded": {
            k: ("leaks the stage label" if v["leakage"]["leaks"] else "not informative")
            for k, v in images.items()
            if not v["kept"]
        },
    }
    return report


def admitted_image_types(report: dict, overrides: dict | None = None) -> list[str]:
    """The photo types a model may use: the feature check's ``image_types_kept`` plus the
    types readmitted for the report's dataset in ``configs/feature_check_overrides.json``
    (PROJECT_PLAN.md D24). Only types the report examined can be readmitted."""
    kept = list(report["decisions"]["image_types_kept"])
    admit = (overrides or {}).get(report.get("dataset", ""), {}).get("admit", {})
    examined = set(report["decisions"]["image_types_kept"]) | set(
        report["decisions"]["image_types_excluded"]
    )
    unknown = set(admit) - examined
    if unknown:
        raise ValueError(
            f"overrides name photo types the report did not examine: {sorted(unknown)}"
        )
    return kept + [k for k in admit if k not in kept]
