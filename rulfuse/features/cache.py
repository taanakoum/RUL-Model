"""Feature caches (PROJECT_PLAN.md §5.8), keyed by signature; resumable.

Signal features, ``interim/features/<dataset>/signal-<encoder>-<key>/``, one row per
distinct segment of the segment store (duplicated recordings are encoded once):

* ``embeddings.npy``: (n, channels, 2·dim) float32, the encoder's embeddings of the
  segment's 512-sample sub-windows, mean- and sd-pooled;
* ``descriptors.npy``: (n, channels, 7) the level descriptors;
* ``spectral.npy``: (n, channels, 11) the rest of the handcrafted set H (B4 only);
* ``done.npy``, ``meta.json`` (signature, shapes, progress).

Image features, ``interim/features/<dataset>/image-<encoder>-<key>/``, one row per row of
the store's image table: ``tokens.npy`` (n, 9, dim) float16, ``done.npy``, ``meta.json``.

Channels are the model channels (measured + magnitude of each 3-axis group); signals are
resampled to the model rate before encoding. A folder whose signature differs from the
expected one is refused, never mixed.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as sps

from rulfuse import paths
from rulfuse.core.channels import (
    DESCRIPTOR_NAMES,
    DESCRIPTORS_VERSION,
    DRIFT_HEAD_SECONDS,
    descriptors,
    expand,
    model_channels,
)
from rulfuse.core.images import normalize, prepared_pixels
from rulfuse.core.schema import SignalSource
from rulfuse.core.segments import subwindow_starts
from rulfuse.core.store import SegmentStore, stable_hash
from rulfuse.features.handcrafted import SPECTRAL_NAMES, spectral_features

FEATURES_VERSION = 1


def features_root() -> Path:
    return paths.interim_dir() / "features"


def resample(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Polyphase resampling along axis 0 with an exact rational ratio (Project 14)."""
    if src_hz == dst_hz:
        return np.asarray(x, dtype=np.float32)
    ratio = Fraction(dst_hz / src_hz).limit_denominator(1000)
    out = sps.resample_poly(np.asarray(x, dtype=np.float64), ratio.numerator, ratio.denominator)
    return out.astype(np.float32)


def _save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def _check_or_create(folder: Path, signature: dict, shapes: dict) -> tuple[dict, bool]:
    meta_path = folder / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["signature"] != signature or meta["shapes"] != shapes:
            raise ValueError(f"{folder} holds other features; delete it to rebuild")
        return meta, False
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"signature": signature, "shapes": shapes, "created": time.strftime("%Y-%m-%d %H:%M")}
    return meta, True


# ----------------------------------------------------------------------------- signals


def signal_signature(store: SegmentStore, encoder_signature: dict) -> dict:
    s = store.settings
    return {
        "features_version": FEATURES_VERSION,
        "kind": "signal",
        "store": store.signature,
        "encoder": encoder_signature,
        "channels": [asdict(c) for c in model_channels(store.channels)],
        "model_rate_hz": s.model_rate_hz,
        "descriptors": {"names": list(DESCRIPTOR_NAMES), "version": DESCRIPTORS_VERSION},
        "spectral": list(SPECTRAL_NAMES),
        "drift_head_s": DRIFT_HEAD_SECONDS,
    }


def signal_folder(store: SegmentStore, encoder_signature: dict, root: Path | None = None) -> Path:
    sig = signal_signature(store, encoder_signature)
    base = Path(root) if root is not None else features_root()
    return base / store.dataset / f"signal-{encoder_signature['encoder']}-{stable_hash(sig)}"


@dataclass
class SignalFeatures:
    folder: Path
    meta: dict
    embeddings: np.ndarray
    descriptors: np.ndarray
    spectral: np.ndarray
    done: np.ndarray

    @classmethod
    def open(cls, folder: str | Path) -> SignalFeatures:
        f = Path(folder)
        meta = json.loads((f / "meta.json").read_text(encoding="utf-8"))
        return cls(
            f,
            meta,
            np.load(f / "embeddings.npy", mmap_mode="r"),
            np.load(f / "descriptors.npy", mmap_mode="r"),
            np.load(f / "spectral.npy", mmap_mode="r"),
            np.load(f / "done.npy"),
        )

    @property
    def complete(self) -> bool:
        return bool(self.done.all())


def distinct_rows(store: SegmentStore) -> np.ndarray:
    """For each distinct segment id, the first segment row that has it."""
    ids = store.segments["distinct_segment"].astype(np.int64)
    n = int(ids.max()) + 1 if len(ids) else 0
    first = np.full(n, -1, dtype=np.int64)
    rows = np.flatnonzero(ids >= 0)
    unique, index = np.unique(ids[rows], return_index=True)
    first[unique] = rows[index]
    return first


def _segment_inputs(
    store: SegmentStore, signals: SignalSource, row: int, cache: dict
) -> tuple[np.ndarray, np.ndarray]:
    """(segment at the model rate (samples, C'), head mean (C',)) of one segment row."""
    s = store.settings
    seg = store.segments
    source = str(seg["source"][row])
    if source not in cache:
        cache.clear()
        cache[source] = np.asarray(signals.read(source))
    array = cache[source]
    channels = store.channels
    rec_start = int(seg["recording_start_sample"][row])
    start, end = int(seg["start_sample"][row]), int(seg["end_sample"][row])
    head_n = max(1, int(DRIFT_HEAD_SECONDS * s.sample_rate_hz))
    head = resample(expand(array[rec_start : rec_start + head_n], channels), s.sample_rate_hz, s.model_rate_hz)
    segment = resample(expand(array[start:end], channels), s.sample_rate_hz, s.model_rate_hz)
    return segment, head.mean(axis=0, dtype=np.float64)


def windows(segment: np.ndarray, length: int) -> np.ndarray:
    """(samples, C) -> (C, k, length) sub-windows covering every sample (edge-padded)."""
    x = np.asarray(segment, dtype=np.float32)
    if len(x) < length:
        x = np.pad(x, ((0, length - len(x)), (0, 0)), mode="edge")
    starts = subwindow_starts(len(x), length)
    return np.stack([x[a : a + length] for a in starts], axis=0).transpose(2, 0, 1)


def extract_signal_features(
    store: SegmentStore,
    signals: SignalSource,
    encoder,
    root: Path | None = None,
    chunk: int = 64,
    limit: int | None = None,
    log=print,
) -> SignalFeatures:
    """Encode every distinct segment not cached yet (``limit``: at most that many now)."""
    encoder_signature = encoder.signature()
    signature = signal_signature(store, encoder_signature)
    folder = signal_folder(store, encoder_signature, root)
    first = distinct_rows(store)
    n, c = len(first), len(model_channels(store.channels))
    shapes = {
        "embeddings": [n, c, 2 * encoder.dim],
        "descriptors": [n, c, len(DESCRIPTOR_NAMES)],
        "spectral": [n, c, len(SPECTRAL_NAMES)],
    }
    meta, new = _check_or_create(folder, signature, shapes)
    if new:
        for name, shape in shapes.items():
            np.lib.format.open_memmap(folder / f"{name}.npy", "w+", np.float32, tuple(shape)).flush()
        _save_npy(folder / "done.npy", np.zeros(n, dtype=bool))
        _write_json(folder / "meta.json", meta)
    arrays = {name: np.load(folder / f"{name}.npy", mmap_mode="r+") for name in shapes}
    done = np.load(folder / "done.npy")
    todo = np.flatnonzero(~done)
    if limit is not None:
        todo = todo[:limit]
    log(f"[features] {store.dataset} {encoder.name}: {len(todo)} segments to encode, {int(done.sum())} cached")
    t0 = time.perf_counter()
    cache: dict = {}
    order = todo[np.argsort(store.segments["source"][first[todo]], kind="stable")]
    for begin in range(0, len(order), chunk):
        ids = order[begin : begin + chunk]
        blocks, counts = [], []
        for d in ids:
            segment, head = _segment_inputs(store, signals, int(first[d]), cache)
            arrays["descriptors"][d] = descriptors(segment, store.settings.model_rate_hz, head)
            arrays["spectral"][d] = spectral_features(segment)
            w = windows(segment, encoder.length)  # (C, k, L)
            blocks.append(w.reshape(-1, encoder.length))
            counts.append(w.shape[1])
        embedded = encoder.encode(np.concatenate(blocks))
        offset = 0
        for d, k in zip(ids, counts, strict=True):
            e = embedded[offset : offset + c * k].reshape(c, k, -1)
            offset += c * k
            arrays["embeddings"][d] = np.concatenate([e.mean(axis=1), e.std(axis=1)], axis=1)
        for array in arrays.values():
            array.flush()
        done[ids] = True
        _save_npy(folder / "done.npy", done)
        meta.update(n_done=int(done.sum()), complete=bool(done.all()), updated=time.strftime("%Y-%m-%d %H:%M"))
        _write_json(folder / "meta.json", meta)
        rate = (begin + len(ids)) / max(time.perf_counter() - t0, 1e-9)
        log(f"[features] {begin + len(ids)}/{len(order)} segments ({rate:.1f}/s)")
    meta.update(n_done=int(done.sum()), complete=bool(done.all()))
    _write_json(folder / "meta.json", meta)
    del arrays
    return SignalFeatures.open(folder)


def open_signal_features(
    store: SegmentStore, encoder_signature: dict, root: Path | None = None
) -> SignalFeatures:
    """The complete cache for this store and encoder signature (refuses anything else)."""
    folder = signal_folder(store, encoder_signature, root)
    features = SignalFeatures.open(folder)
    if features.meta["signature"] != signal_signature(store, encoder_signature):
        raise ValueError(f"{folder}: signature mismatch")
    if not features.complete:
        raise ValueError(f"{folder}: incomplete ({int(features.done.sum())}/{len(features.done)})")
    return features


def find_features(store: SegmentStore, kind: str, encoder: str, root: Path | None = None) -> Path:
    """The single complete cache of ``kind`` ("signal"/"image") and ``encoder`` built on this
    store (the encoder's revision is read from the cache, so no weights are needed)."""
    base = (Path(root) if root is not None else features_root()) / store.dataset
    matches = []
    for meta_path in base.glob(f"{kind}-{encoder}-*/meta.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["signature"]["store"] == store.signature and meta.get("complete"):
            matches.append(meta_path.parent)
    if len(matches) != 1:
        raise FileNotFoundError(f"{len(matches)} complete {kind}-{encoder} caches for {store.signature} in {base}")
    return matches[0]


# ----------------------------------------------------------------------------- images


def image_signature(store: SegmentStore, encoder_signature: dict) -> dict:
    return {
        "features_version": FEATURES_VERSION,
        "kind": "image",
        "store": store.signature,
        "encoder": encoder_signature,
        "image_specs": {k: asdict(v) for k, v in sorted(store.image_specs.items())},
    }


def image_folder(store: SegmentStore, encoder_signature: dict, root: Path | None = None) -> Path:
    sig = image_signature(store, encoder_signature)
    base = Path(root) if root is not None else features_root()
    return base / store.dataset / f"image-{encoder_signature['encoder']}-{stable_hash(sig)}"


@dataclass
class ImageFeatures:
    folder: Path
    meta: dict
    tokens: np.ndarray  # (n_images, n_tokens, dim), rows = the store's image rows
    done: np.ndarray

    @classmethod
    def open(cls, folder: str | Path) -> ImageFeatures:
        f = Path(folder)
        meta = json.loads((f / "meta.json").read_text(encoding="utf-8"))
        return cls(f, meta, np.load(f / "tokens.npy", mmap_mode="r"), np.load(f / "done.npy"))

    @property
    def complete(self) -> bool:
        return bool(self.done.all())


def extract_image_features(
    store: SegmentStore,
    encoder,
    root: Path | None = None,
    kinds: list[str] | None = None,
    batch: int = 16,
    limit: int | None = None,
    log=print,
) -> ImageFeatures:
    encoder_signature = encoder.signature()
    signature = image_signature(store, encoder_signature)
    folder = image_folder(store, encoder_signature, root)
    n = len(store.images["unit"])
    shapes = {"tokens": [n, encoder.n_tokens, encoder.dim]}
    meta, new = _check_or_create(folder, signature, shapes)
    if new:
        np.lib.format.open_memmap(folder / "tokens.npy", "w+", np.float16, tuple(shapes["tokens"])).flush()
        _save_npy(folder / "done.npy", np.zeros(n, dtype=bool))
        _write_json(folder / "meta.json", meta)
    tokens = np.load(folder / "tokens.npy", mmap_mode="r+")
    done = np.load(folder / "done.npy")
    specs = store.image_specs
    t0, encoded = time.perf_counter(), 0
    for kind in kinds or sorted(specs):
        todo = np.flatnonzero(~done & (store.images["kind"] == kind))
        if limit is not None:
            todo = todo[: max(0, limit - encoded)]
        log(f"[features] {store.dataset} {encoder.name} {kind}: {len(todo)} images to encode")
        for begin in range(0, len(todo), batch):
            rows = todo[begin : begin + batch]
            pixels = np.stack([normalize(prepared_pixels(store.image_path(r), specs[kind])) for r in rows])
            tokens[rows] = encoder.encode(pixels).astype(np.float16)
            tokens.flush()
            done[rows] = True
            _save_npy(folder / "done.npy", done)
            encoded += len(rows)
            meta.update(n_done=int(done.sum()), complete=bool(done.all()), updated=time.strftime("%Y-%m-%d %H:%M"))
            _write_json(folder / "meta.json", meta)
            log(f"[features] {kind}: {begin + len(rows)}/{len(todo)} ({encoded / max(time.perf_counter() - t0, 1e-9):.2f} images/s)")
    meta.update(n_done=int(done.sum()), complete=bool(done.all()))
    _write_json(folder / "meta.json", meta)
    del tokens
    return ImageFeatures.open(folder)
