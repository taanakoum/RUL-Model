"""Fold-level normalization and the causal unit baseline (PROJECT_PLAN.md §5.9, part P7).

* ``Scaler`` (and ``PCA``, kept for baseline B4 and analysis only, D17) are fitted on the
  training units' rows (``train_rows``); validation and test units are only transformed.
* ``causal_baseline`` gives every token the mean of the unit's baseline tokens *seen so
  far* (its first B recordings), so tokens become relative to the unit's healthy start
  without looking ahead. ``relative_to_first`` does the same with only the first valid
  token (images: relative to the unit's first image of that type).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rulfuse.core.folds import Fold

SCALE_FLOOR = 1e-8


def train_rows(units: Sequence[str] | np.ndarray, fold: Fold) -> np.ndarray:
    """Boolean mask of the rows that belong to the fold's training units."""
    return np.isin(np.asarray(units), np.asarray(fold.train))


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> Scaler:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Scaler.fit needs a non-empty (rows, features) array")
        scale = x.std(axis=0)
        scale[scale < SCALE_FLOOR] = 1.0
        return cls(x.mean(axis=0), scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.scale


@dataclass(frozen=True)
class PCA:
    mean: np.ndarray
    components: np.ndarray  # (k, d), orthonormal rows
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, n_components: int) -> PCA:
        x = np.asarray(x, dtype=np.float64)
        k = min(n_components, x.shape[0], x.shape[1])
        if k < 1:
            raise ValueError("PCA needs at least one component, row and feature")
        mean = x.mean(axis=0)
        _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
        components = vt[:k].copy()
        # Deterministic signs: the largest-magnitude loading of each component is positive.
        signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
        components *= np.where(signs == 0, 1.0, signs)[:, None]
        variance = singular**2
        ratio = variance[:k] / variance.sum() if variance.sum() > 0 else np.zeros(k)
        return cls(mean, components, ratio)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.components.T


@dataclass(frozen=True)
class BlockTransform:
    """Standardize + PCA per block of input columns; outputs concatenated in block order
    (baseline B4 and analysis only)."""

    names: tuple[str, ...]
    columns: tuple[np.ndarray, ...]
    scalers: tuple[Scaler, ...]
    pcas: tuple[PCA, ...]

    @classmethod
    def fit(
        cls, x: np.ndarray, blocks: Mapping[str, Sequence[int]], n_components: int
    ) -> BlockTransform:
        names, columns, scalers, pcas = [], [], [], []
        for name, cols in blocks.items():
            index = np.asarray(cols, dtype=np.int64)
            scaler = Scaler.fit(x[:, index])
            names.append(name)
            columns.append(index)
            scalers.append(scaler)
            pcas.append(PCA.fit(scaler.transform(x[:, index]), n_components))
        return cls(tuple(names), tuple(columns), tuple(scalers), tuple(pcas))

    def transform(self, x: np.ndarray) -> np.ndarray:
        parts = [
            pca.transform(scaler.transform(x[:, cols]))
            for cols, scaler, pca in zip(self.columns, self.scalers, self.pcas, strict=True)
        ]
        return np.concatenate(parts, axis=1)

    def output_slices(self) -> dict[str, slice]:
        out, start = {}, 0
        for name, pca in zip(self.names, self.pcas, strict=True):
            width = pca.components.shape[0]
            out[name] = slice(start, start + width)
            start += width
        return out


def unit_shift(x: np.ndarray, units: Sequence[str] | np.ndarray, fold: Fold) -> np.ndarray:
    """Per feature, the fold's test unit's mean in the training units' standardization
    (0: no shift). The median absolute value is the P7 report quantity."""
    units = np.asarray(units)
    scaler = Scaler.fit(np.asarray(x)[train_rows(units, fold)])
    return scaler.transform(np.asarray(x)[units == fold.test]).mean(axis=0)


def causal_baseline(
    values: np.ndarray, in_baseline: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    """Row i gets the mean of ``values[j]`` over rows j <= i that are in the baseline (and
    valid). Rows must be one unit's tokens in time order. Rows before the first baseline
    row get 0 (no baseline seen yet)."""
    values = np.asarray(values, dtype=np.float64)
    use = np.asarray(in_baseline, dtype=bool)
    if valid is not None:
        use = use & np.asarray(valid, dtype=bool)
    weights = use.astype(np.float64)[:, None]
    sums = np.cumsum(values * weights, axis=0)
    counts = np.cumsum(weights, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        baseline = np.where(counts > 0, sums / np.maximum(counts, 1.0), 0.0)
    return baseline


def relative_to_baseline(
    values: np.ndarray, in_baseline: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) - causal_baseline(values, in_baseline, valid)


def relative_to_first(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Each row minus the first valid row (rows before it are 0)."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.ones(len(values), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    out = np.zeros_like(values)
    if ok.any():
        start = int(np.argmax(ok))
        out[start:] = values[start:] - values[start]
    return out
