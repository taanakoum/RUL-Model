"""The contract between a dataset adapter and the core (PROJECT_PLAN.md D18, §6.11).

An adapter turns a dataset into a :class:`Dataset`: a list of :class:`Unit` objects (here a
tool) plus the settings and the signal source. Each unit gives:

* **recordings**: contiguous sensor captures, each with its operating start time (seconds
  of operation, gaps of unrecorded operation included), its duration and, when the signal
  exists, a :class:`SignalRef` into the dataset's :class:`SignalSource`;
* **images**: each with a type and the operating time after which it may be read;
* an optional **health indicator** (HI) per recording, measured at the recording's end;
* optional **stage labels** and acquisition **groups**, used only by the leakage probe;
* the **failure time** in seconds of operation (None: censored).

Everything an adapter knows beyond that (here passes, runs, blocks, image labels) goes into
``attrs``. The core passes ``attrs`` through to its tables but never reads it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

AXES = ("x", "y", "z", "none")
DERIVATIONS = ("raw", "magnitude")
TIME_TOLERANCE_S = 1e-6


class SchemaError(ValueError):
    """An adapter's output breaks the contract."""


@dataclass(frozen=True)
class ChannelSpec:
    """One signal channel, described so that a model can tell channels apart without
    knowing the dataset (the channel-descriptor embedding of §6.3)."""

    name: str
    sensor: str  # e.g. "force", "acceleration", "sound"
    axis: str = "none"
    derivation: str = "raw"
    unit: str = ""


@dataclass(frozen=True)
class SignalRef:
    """Samples [start, end) of one source signal (untrimmed rows of ``SignalSource.read``)."""

    source: str
    start: int
    end: int

    @property
    def samples(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Recording:
    key: str  # unique within the dataset
    start_s: float  # operating time at the recording's start
    duration_s: float
    signal: SignalRef | None = None  # None: the signal is missing (a masked token)
    hi: float = math.nan  # HI at the recording's end (a target: may be smoothed with the future)
    hi_causal: float = math.nan  # the same, smoothed with the past only (oracle inputs)
    stage: int | None = None  # optional ordinal stage label (leakage probe only)
    group: str = ""  # acquisition group, e.g. one setup (leakage probe only)
    attrs: Mapping[str, object] = field(default_factory=dict)  # never an input

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class Image:
    key: str  # unique within (unit, kind)
    kind: str  # image type, a key of ``Dataset.image_specs``
    path: str  # absolute, or relative to ``Dataset.image_root``
    available_s: float  # readable only by predictions made strictly after this time


@dataclass(frozen=True)
class Unit:
    name: str
    recordings: tuple[Recording, ...]
    images: tuple[Image, ...] = ()
    failure_s: float | None = None  # seconds of operation; None: censored
    attrs: Mapping[str, object] = field(default_factory=dict)  # never an input


@dataclass(frozen=True)
class ImageSpec:
    """How the images of one type are prepared before any encoder (§5.7)."""

    height: int
    width: int
    crop: tuple[float, float, float, float] | None = None  # top, left, bottom, right fractions
    standardize: bool = False  # per-image luminance standardization


@dataclass(frozen=True)
class Settings:
    horizon_s: float  # H: cap of the training target and the evaluation window
    sample_rate_hz: float  # rate of the source signals
    model_rate_hz: float = 1000.0  # rate the encoders and descriptors see
    baseline_recordings: int = 4  # B: the unit's healthy start
    segment_s: float = 10.0  # S
    min_remainder_s: float = 5.0  # a shorter remainder joins the previous segment
    hi_name: str = ""  # e.g. "flank wear (um)"; empty when the dataset has no HI


class SignalSource(Protocol):
    def read(self, source: str) -> np.ndarray:
        """(samples, channels) float32 of one source signal, untrimmed."""
        ...


@dataclass(frozen=True)
class ArraySignals:
    """A :class:`SignalSource` held in memory (synthetic adapters, tests)."""

    arrays: Mapping[str, np.ndarray]

    def read(self, source: str) -> np.ndarray:
        return self.arrays[source]


@dataclass(frozen=True)
class Dataset:
    name: str
    units: tuple[Unit, ...]
    channels: tuple[ChannelSpec, ...]  # measured channels, in the source's column order
    settings: Settings
    signals: SignalSource
    image_specs: Mapping[str, ImageSpec] = field(default_factory=dict)
    image_root: Path | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)  # hashed into signatures

    @property
    def unit_names(self) -> tuple[str, ...]:
        return tuple(unit.name for unit in self.units)

    def unit(self, name: str) -> Unit:
        for unit in self.units:
            if unit.name == name:
                return unit
        raise KeyError(name)

    def image_path(self, image: Image) -> Path:
        path = Path(image.path)
        if path.is_absolute() or self.image_root is None:
            return path
        return self.image_root / path


def _check_unique(names: Sequence[str], what: str, problems: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if not name:
            problems.append(f"empty {what} name")
        elif name in seen:
            problems.append(f"duplicate {what} {name!r}")
        seen.add(name)


def validate(ds: Dataset, check_signals: bool = False) -> None:
    """Raise :class:`SchemaError` listing every broken rule. ``check_signals`` also reads
    each referenced source once (shape and finiteness)."""
    problems: list[str] = []
    s = ds.settings
    if not s.horizon_s > 0:
        problems.append("settings: horizon_s must be > 0")
    if not (s.sample_rate_hz > 0 and s.model_rate_hz > 0):
        problems.append("settings: sample rates must be > 0")
    if s.baseline_recordings < 1:
        problems.append("settings: baseline_recordings must be >= 1")
    if not (s.segment_s > 0 and 0 <= s.min_remainder_s <= s.segment_s):
        problems.append("settings: need segment_s > 0 and 0 <= min_remainder_s <= segment_s")
    _check_unique([c.name for c in ds.channels], "channel", problems)
    for channel in ds.channels:
        if channel.axis not in AXES:
            problems.append(f"channel {channel.name}: axis must be one of {AXES}")
        if channel.derivation not in DERIVATIONS:
            problems.append(f"channel {channel.name}: derivation must be one of {DERIVATIONS}")
    if not ds.channels:
        problems.append("a dataset needs at least one channel")
    _check_unique(list(ds.unit_names), "unit", problems)
    _check_unique([r.key for u in ds.units for r in u.recordings], "recording", problems)
    sources: dict[str, int] = {}
    for unit in ds.units:
        where = f"unit {unit.name}"
        if not unit.recordings:
            problems.append(f"{where}: no recordings")
            continue
        previous_end = -math.inf
        for rec in unit.recordings:
            if not (math.isfinite(rec.start_s) and rec.duration_s > 0):
                problems.append(f"{where}, {rec.key}: need a finite start and duration > 0")
            if rec.start_s < previous_end - TIME_TOLERANCE_S:
                problems.append(f"{where}, {rec.key}: starts before the previous recording ends")
            previous_end = rec.end_s
            if rec.signal is not None:
                ref = rec.signal
                if not 0 <= ref.start < ref.end:
                    problems.append(f"{where}, {rec.key}: signal bounds must be 0 <= start < end")
                elif abs(ref.samples / s.sample_rate_hz - rec.duration_s) > 1.0 / s.sample_rate_hz:
                    problems.append(f"{where}, {rec.key}: duration disagrees with its samples")
                sources[ref.source] = max(sources.get(ref.source, 0), ref.end)
        _check_unique([f"{im.kind}/{im.key}" for im in unit.images], f"{where} image", problems)
        for image in unit.images:
            if image.kind not in ds.image_specs:
                problems.append(f"{where}: image type {image.kind!r} has no ImageSpec")
            if not math.isfinite(image.available_s):
                problems.append(f"{where}: image {image.key} needs a finite availability time")
        if unit.failure_s is not None:
            last_end = unit.recordings[-1].end_s
            first_end = unit.recordings[0].end_s
            if not first_end - TIME_TOLERANCE_S <= unit.failure_s <= last_end + TIME_TOLERANCE_S:
                problems.append(f"{where}: failure time outside its recordings")
    if check_signals:
        for source, needed in sources.items():
            array = np.asarray(ds.signals.read(source))
            if array.ndim != 2 or array.shape[1] != len(ds.channels):
                problems.append(
                    f"signal {source}: shape {array.shape}, need (n, {len(ds.channels)})"
                )
            elif len(array) < needed:
                problems.append(f"signal {source}: {len(array)} samples, references need {needed}")
            elif not np.isfinite(array).all():
                problems.append(f"signal {source}: non-finite values")
    if problems:
        raise SchemaError(f"{ds.name}: {len(problems)} problem(s): " + "; ".join(problems[:20]))
