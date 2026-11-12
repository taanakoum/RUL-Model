"""Per-unit token inputs for one fold (PROJECT_PLAN.md §5.9, §6.3, §7.2 steps 2-4; part P10).

For every unit the fold sees, :func:`build_fold_data` returns a :class:`UnitData`. It holds
the un-augmented inputs in time order, as NumPy arrays. :mod:`rulfuse.sequences` then turns
a unit (augmented or not) into model sequences.

**Signal tokens.** There is one per segment row, including segments after the failure:
they are history for nothing, but they carry HI targets. Each channel has 4 blocks:

* ``emb``: the encoder embedding, layer-normalized per vector by a fixed function (no fitted
  statistic, D17);
* ``desc``: the 7 level descriptors, standardized on the training units' segments (the
  only fitted input statistic, §5.9);
* the change of both since the unit's healthy start: the causal mean over the baseline
  segments seen so far (``in_baseline``). Before any baseline segment the change is 0.

The token mode keeps the absolute blocks, the relative ones, or both. A segment without a
signal has ``present = False`` and zero features; the model puts a learned mask vector there.

**Image tokens.** There is one per image of an admitted kind:

* the encoder's 9 tokens, layer-normalized, and their change from the unit's first image
  of that kind;
* the image's HI readout (when the dataset has an HI): a ridge from the image vector to the
  HI. It is cross-fitted, so a training unit's readout never uses its own labels (stacking
  leakage, §7.2 step 3). It is standardized with the training units' HI scale.

An image belongs to the last recording that ended at or before its availability time
(``recording``); augmentation moves it with that recording.

Nothing here refers to a dataset: units, recordings, segments, image kinds and an optional
HI are the generic schema (D18).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from rulfuse.config import FitConfig
from rulfuse.core.folds import Fold
from rulfuse.core.normalization import Scaler
from rulfuse.core.schema import TIME_TOLERANCE_S
from rulfuse.core.store import SegmentStore, stable_hash
from rulfuse.features.cache import ImageFeatures, SignalFeatures
from rulfuse.features.probe import cross_fitted_readout

LN_EPS = 1e-5
READOUT_VERSION = 1


def cached_readout(
    store: SegmentStore, image_features, kind: str, fold: Fold, compute
) -> np.ndarray:
    """The cross-fitted HI readout of one image kind for one fold, cached next to the store
    (A3, 2026-11-12). It depends only on the store, the image cache, the kind and the training
    units, so every arm and seed of a fold reuses it; the values are saved in float64, so a
    cached readout equals a fresh one exactly."""
    key = stable_hash(
        {
            "version": READOUT_VERSION,
            "store": store.signature,
            "image_cache": Path(image_features.folder).name,
            "kind": kind,
            "train": sorted(fold.train),
        }
    )
    path = store.root / "readouts" / f"{kind}-{key}.npy"
    if path.is_file():
        return np.load(path)
    out = np.asarray(compute(), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, out)
    os.replace(tmp, path)
    return out


def layer_norm(x: np.ndarray) -> np.ndarray:
    """Fixed (unfitted) layer normalization over the last axis."""
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(sd**2 + LN_EPS)


def causal_change(values: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Each row minus the mean of the rows j <= i flagged ``use``; 0 before the first one.
    Rows are one unit's tokens in time order; works on any trailing shape."""
    values = np.asarray(values, dtype=np.float32)
    w = np.asarray(use, dtype=np.float32).reshape((-1,) + (1,) * (values.ndim - 1))
    sums = np.cumsum(values * w, axis=0)
    counts = np.cumsum(w, axis=0)
    seen = counts > 0
    baseline = np.where(seen, sums / np.maximum(counts, 1.0), 0.0)
    return np.where(seen, values - baseline, 0.0).astype(np.float32)


@dataclass
class UnitData:
    name: str
    horizon_s: float
    failure_s: float | None
    # recordings (time order)
    rec_start: np.ndarray  # (R,)
    rec_end: np.ndarray  # (R,)
    rec_baseline: np.ndarray  # (R,) bool
    rec_hi_causal: np.ndarray  # (R,) standardized causal HI (nan when absent)
    # signal tokens (one per segment, time order)
    seg_rec: np.ndarray  # (S,) recording position 0..R-1
    seg_elapsed: np.ndarray  # (S,) seconds into the recording
    seg_present: np.ndarray  # (S,) bool
    seg_hi: np.ndarray  # (S,) standardized HI target (nan when absent)
    signal: np.ndarray  # (S, C, F) float32
    # image tokens
    img_rec: np.ndarray  # (I,) recording position the image follows
    img_offset: np.ndarray  # (I,) availability time minus that recording's end
    img_kind: np.ndarray  # (I,) index into the admitted kinds
    images: np.ndarray  # (I, P, G) float32
    img_readout: np.ndarray  # (I,) standardized HI readout (0 without an HI)

    @cached_property
    def rec_segments(self) -> list[np.ndarray]:
        """Per recording position, its signal-token rows (computed once; A2)."""
        return [np.flatnonzero(self.seg_rec == p) for p in range(len(self.rec_start))]

    @cached_property
    def rec_images(self) -> list[np.ndarray]:
        """Per recording position, the image rows that follow it (computed once; A2)."""
        return [np.flatnonzero(self.img_rec == p) for p in range(len(self.rec_start))]

    @property
    def failure_rec(self) -> int:
        """Position of the recording that contains the failure (R if censored)."""
        if self.failure_s is None:
            return len(self.rec_start)
        inside = np.flatnonzero(
            (self.rec_start < self.failure_s) & (self.failure_s <= self.rec_end + TIME_TOLERANCE_S)
        )
        return int(inside[0]) if len(inside) else len(self.rec_start)


@dataclass
class FoldData:
    fold: Fold
    units: dict[str, UnitData]
    channels: list[tuple[str, str, str]]  # (sensor, axis, derivation) per model channel
    kinds: tuple[str, ...]
    signal_dim: int
    image_dim: int
    image_tokens: int
    has_hi: bool
    hi_scale: tuple[float, float]  # mean, sd of the training units' HI
    transforms: dict


def _blocks(absolute: list[np.ndarray], relative: list[np.ndarray], mode: str) -> np.ndarray:
    parts = (absolute if mode in ("both", "absolute") else []) + (
        relative if mode in ("both", "relative") else []
    )
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _image_recordings(store: SegmentStore, unit: str, available: np.ndarray) -> np.ndarray:
    rows = store.recording_rows(unit)
    ends = store.recordings["end_s"][rows].astype(float)
    pos = np.searchsorted(ends, available + TIME_TOLERANCE_S, side="right") - 1
    return np.maximum(pos, 0)


def build_fold_data(
    store: SegmentStore,
    signal_features: SignalFeatures | None,
    image_features: ImageFeatures | None,
    fold: Fold,
    config: FitConfig,
    units: list[str] | None = None,
) -> FoldData:
    """Tokens of the fold's units (training, validation and test by default). Statistics come
    from the training units only (P7)."""
    seg, rec, img = store.segments, store.recordings, store.images
    horizon = store.settings.horizon_s
    names = units or [*fold.train, fold.validation, fold.test]
    names = list(dict.fromkeys(names))
    train = np.isin(seg["unit"], fold.train)

    # HI scale from the training units' recordings
    rec_train = np.isin(rec["unit"], fold.train)
    hi_all = rec["hi"].astype(float)
    hi_ok = rec_train & np.isfinite(hi_all)
    has_hi = bool(hi_ok.any())
    hi_mean = float(hi_all[hi_ok].mean()) if has_hi else 0.0
    hi_sd = float(hi_all[hi_ok].std()) if has_hi else 1.0
    hi_sd = hi_sd if hi_sd > 0 else 1.0

    # signal: descriptor standardization on the training units' present segments
    present = seg["signal_present"].astype(bool) & (seg["distinct_segment"] >= 0)
    ids = np.maximum(seg["distinct_segment"].astype(np.int64), 0)
    if signal_features is not None:
        n_ch = signal_features.descriptors.shape[1]
        desc_all = np.asarray(signal_features.descriptors, dtype=np.float32)[ids]
        desc_all[~present] = 0.0
        scaler = Scaler.fit(desc_all[train & present].reshape(int((train & present).sum()), -1))
        emb_all = layer_norm(np.asarray(signal_features.embeddings, dtype=np.float32)[ids])
        emb_all[~present] = 0.0
        desc_std = scaler.transform(desc_all.reshape(len(ids), -1)).reshape(desc_all.shape)
        desc_std = desc_std.astype(np.float32)
        desc_std[~present] = 0.0
        channel_dim = int(emb_all.shape[-1]) + int(desc_all.shape[-1])
        signal_dim = channel_dim * (2 if config.token_mode == "both" else 1)
        meta_channels = signal_features.meta["signature"]["channels"]
        channels = [(c["sensor"], c["axis"], c["derivation"]) for c in meta_channels]
        transforms = {
            "descriptor_mean": scaler.mean.tolist(),
            "descriptor_scale": scaler.scale.tolist(),
        }
    else:
        n_ch, signal_dim, channels, transforms = 1, 1, [("none", "none", "none")], {}
        emb_all = desc_std = None

    # images: tokens and the cross-fitted HI readout
    kinds = config.image_kinds
    image_dim, image_tokens = 1, 1
    readout = np.zeros(len(img["unit"]))
    if kinds and image_features is not None:
        image_tokens = image_features.tokens.shape[1]
        token_dim = image_features.tokens.shape[2]
        image_dim = token_dim * (2 if config.token_mode == "both" else 1)
        for kind in kinds:
            rows = store.image_rows(kind=kind)
            if not len(rows):
                raise ValueError(f"no images of kind {kind!r} in the store")
            if has_hi:
                tokens = np.asarray(image_features.tokens[rows], dtype=np.float32)
                vectors = np.concatenate([tokens[:, 0], tokens[:, 1:].mean(axis=1)], axis=1)
                target = np.full(len(rows), np.nan)
                for i, r in enumerate(rows):
                    unit = str(img["unit"][r])
                    pos = _image_recordings(store, unit, np.array([img["available_s"][r]]))[0]
                    target[i] = hi_all[store.recording_rows(unit)[pos]]
                groups = img["unit"][rows].astype(str)
                out = cached_readout(
                    store,
                    image_features,
                    kind,
                    fold,
                    lambda v=vectors, t=target, g=groups: cross_fitted_readout(v, t, g, fold.train),
                )
                readout[rows] = (out - hi_mean) / hi_sd

    data: dict[str, UnitData] = {}
    for name in names:
        rrows = store.recording_rows(name)
        srows = store.segment_rows(name)
        rec_index = {int(i): p for p, i in enumerate(rec["index"][rrows])}
        seg_rec = np.array([rec_index[int(i)] for i in seg["recording"][srows]], dtype=np.int64)
        seg_present = present[srows]
        hi_seg = (seg["hi"][srows].astype(float) - hi_mean) / hi_sd
        if emb_all is not None and desc_std is not None:
            use = seg["in_baseline"][srows].astype(bool) & seg_present
            emb, desc = emb_all[srows], desc_std[srows]
            signal = _blocks(
                [emb, desc], [causal_change(emb, use), causal_change(desc, use)], config.token_mode
            )
            signal[~seg_present] = 0.0
        else:
            signal = np.zeros((len(srows), n_ch, 1), dtype=np.float32)
        hi_causal = (rec["hi_causal"][rrows].astype(float) - hi_mean) / hi_sd
        ends = rec["end_s"][rrows].astype(float)
        if kinds and image_features is not None:
            irows = np.concatenate([store.image_rows(name, kind) for kind in kinds])
            avail = img["available_s"][irows].astype(float)
            order = np.argsort(avail, kind="stable")
            irows, avail = irows[order], avail[order]
            kind_index = np.array([kinds.index(str(k)) for k in img["kind"][irows]], dtype=np.int64)
            tok = layer_norm(np.asarray(image_features.tokens[irows], dtype=np.float32))
            rel = np.zeros_like(tok)
            for k in range(len(kinds)):
                mine = kind_index == k
                if mine.any():
                    rel[mine] = tok[mine] - tok[mine][:1]
            images = _blocks([tok], [rel], config.token_mode)
            img_rec = _image_recordings(store, name, avail)
            img_offset = avail - ends[img_rec]
            img_readout = readout[irows].astype(np.float32)
        else:
            images = np.zeros((0, image_tokens, image_dim), dtype=np.float32)
            img_rec = np.zeros(0, dtype=np.int64)
            img_offset = np.zeros(0)
            kind_index = np.zeros(0, dtype=np.int64)
            img_readout = np.zeros(0, dtype=np.float32)
        failure = store.units[name]["failure_s"]
        data[name] = UnitData(
            name=name,
            horizon_s=horizon,
            failure_s=None if failure is None else float(failure),
            rec_start=rec["start_s"][rrows].astype(float),
            rec_end=ends,
            rec_baseline=rec["in_baseline"][rrows].astype(bool),
            rec_hi_causal=hi_causal,
            seg_rec=seg_rec,
            seg_elapsed=seg["elapsed_s"][srows].astype(float),
            seg_present=seg_present,
            seg_hi=hi_seg,
            signal=signal,
            img_rec=img_rec.astype(np.int64),
            img_offset=img_offset.astype(float),
            img_kind=kind_index,
            images=images,
            img_readout=img_readout,
        )
    return FoldData(
        fold=fold,
        units=data,
        channels=channels,
        kinds=kinds,
        signal_dim=signal_dim,
        image_dim=image_dim,
        image_tokens=image_tokens,
        has_hi=has_hi,
        hi_scale=(hi_mean, hi_sd),
        transforms=transforms,
    )
