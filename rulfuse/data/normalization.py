"""Fold-level normalization and the causal tool baseline (PROJECT_PLAN.md §5.9, part P7).

* ``Scaler`` and ``PCA`` are fitted on the training tools' rows only (``train_rows``); the
  validation and test tools are transformed, never fitted on.
* ``BlockTransform`` standardizes and PCA-reduces each block of columns separately (one
  block per force channel, §5.4) and concatenates the outputs, so a whole channel can be
  masked by zeroing its output slice.
* ``causal_baseline`` gives every token the mean of the tool's baseline tokens *seen so
  far* (its first B recorded passes), so tokens become relative to the tool's healthy
  start without looking ahead. ``relative_to_first`` does the same with only the first
  valid token (photos: relative to the tool's first photo).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rulfuse.data.folds import ToolFold

SCALE_FLOOR = 1e-8


def train_rows(tools: Sequence[str] | np.ndarray, fold: ToolFold) -> np.ndarray:
    """Boolean mask of the rows that belong to the fold's training tools."""
    return np.isin(np.asarray(tools), np.asarray(fold.train_tools))


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
    """Standardize + PCA per block of input columns; outputs concatenated in block order."""

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
        """Output columns of each block (for masking one channel)."""
        out, start = {}, 0
        for name, pca in zip(self.names, self.pcas, strict=True):
            width = pca.components.shape[0]
            out[name] = slice(start, start + width)
            start += width
        return out


def tool_shift(x: np.ndarray, tools: Sequence[str] | np.ndarray, fold: ToolFold) -> np.ndarray:
    """Per feature, the fold's test tool's mean in the training tools' standardization.

    0 means no shift. OmniFuser's EDA (§8) measured 0.16-0.32 sd (median absolute, over the
    three raw force channels) and +1.15 sd for T1's Fx. The median absolute value of this
    vector is the P7 acceptance quantity.
    """
    tools = np.asarray(tools)
    scaler = Scaler.fit(np.asarray(x)[train_rows(tools, fold)])
    return scaler.transform(np.asarray(x)[tools == fold.test]).mean(axis=0)


def causal_baseline(
    values: np.ndarray, in_baseline: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    """Row i gets the mean of ``values[j]`` over rows j <= i that are in the baseline (and
    valid). Rows must be one tool's tokens in time order. Rows before the first baseline row
    get 0 (no baseline seen yet)."""
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


def baseline_rows(
    positions: np.ndarray, recorded_positions: Sequence[int], passes: int = 4
) -> np.ndarray:
    """Rows whose pass is among the tool's first ``passes`` recorded passes."""
    first = sorted(recorded_positions)[:passes]
    return np.isin(np.asarray(positions), np.asarray(first))


def relative_to_baseline(
    values: np.ndarray, in_baseline: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) - causal_baseline(values, in_baseline, valid)


def relative_to_first(values: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Each row minus the first valid row (rows before it are 0): photos against the tool's
    first photo."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.ones(len(values), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    out = np.zeros_like(values)
    if ok.any():
        start = int(np.argmax(ok))
        out[start:] = values[start:] - values[start]
    return out
