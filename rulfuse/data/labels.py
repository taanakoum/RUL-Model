"""Labels: time to failure, prediction points and RUL targets (PROJECT_PLAN.md D2, D3, §3, §5.6).

TTF rules (§2.3), applied to a tool's *recorded* label sequence in (run, block) order:

* ``confirmed`` (D2, default): the first ``dulled`` pass followed by at least 2 more
  ``dulled`` passes among its next 3 recorded passes;
* ``literal``: the first ``dulled`` pass;
* ``wear110``: the first pass whose isotonic-smoothed flank wear reaches 110 µm;
* ``tool_label3``: the first pass with ``tool_label`` 3;
* ``dulled_onwards``: the first pass after which every pass is ``dulled``.

Failure is at the end of the TTF pass. Time has two units: passes (label time
t = (p − 1) + e / D_p, which uses the pass duration D_p and is therefore a label, never an
input) and seconds of cutting (tau).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression

from rulfuse.data.nonastreda import PassMeta
from rulfuse.data.segments import MIN_REMAINDER_SECONDS, SEGMENT_SECONDS, segment_ends
from rulfuse.data.timeline import Slot

TTF_RULES = ("confirmed", "literal", "wear110", "tool_label3", "dulled_onwards")
WEAR_THRESHOLD_UM = 110.0
K_PASSES = 15.0
K_SECONDS = 1500.0
CONFIRM_FOLLOW = 3
CONFIRM_NEED = 2


def isotonic(values: npt.ArrayLike) -> np.ndarray:
    """Non-decreasing least-squares fit over the whole sequence (a target, uses the future)."""
    return np.asarray(isotonic_regression(np.asarray(values, dtype=np.float64)).x)


def isotonic_causal(values: npt.ArrayLike) -> np.ndarray:
    """Value i is the last point of the isotonic fit to values[: i + 1] (uses only the past)."""
    array = np.asarray(values, dtype=np.float64)
    return np.asarray([isotonic(array[: index + 1])[-1] for index in range(len(array))])


def ttf_index(
    rule: str,
    metas: Sequence[PassMeta],
    *,
    follow: int = CONFIRM_FOLLOW,
    need: int = CONFIRM_NEED,
    threshold_um: float = WEAR_THRESHOLD_UM,
) -> int | None:
    """Index, in the recorded sequence ``metas`` of one tool, of its TTF pass (None: none)."""
    labels = [meta.image_label for meta in metas]
    if rule == "confirmed":
        for index, label in enumerate(labels):
            later = labels[index + 1 : index + 1 + follow]
            if label == "dulled" and sum(item == "dulled" for item in later) >= need:
                return index
        return None
    if rule == "literal":
        return next((i for i, label in enumerate(labels) if label == "dulled"), None)
    if rule == "wear110":
        smoothed = isotonic([meta.flank_wear for meta in metas])
        return next((i for i, value in enumerate(smoothed) if value >= threshold_um), None)
    if rule == "tool_label3":
        return next((i for i, meta in enumerate(metas) if meta.tool_label == 3), None)
    if rule == "dulled_onwards":
        if not labels or labels[-1] != "dulled":
            return None
        not_dulled = [i for i, label in enumerate(labels) if label != "dulled"]
        return not_dulled[-1] + 1 if not_dulled else 0
    raise ValueError(f"TTF rule must be one of {TTF_RULES}")


@dataclass(frozen=True)
class Points:
    """Prediction points of one tool: every segment end of its recorded passes up to the TTF."""

    tool: str
    position: np.ndarray  # pass position on the axis (1-based)
    segment: np.ndarray  # segment index inside the pass (1-based)
    elapsed_in_pass: np.ndarray  # e, seconds since the trimmed start of the pass (input)
    duration: np.ndarray  # D_p of the pass (label use only)
    t_label: np.ndarray  # (p - 1) + e / D_p, passes (label use only)
    tau: np.ndarray  # cumulative cutting seconds (input)
    completed_passes: np.ndarray  # p - 1 (input)
    mean_past_duration: np.ndarray  # mean D of the tool's recorded passes before p (NaN: none)
    force_present: np.ndarray  # bool
    record_id: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.position)


def prediction_points(
    timeline: Sequence[Slot],
    last_position: int,
    segment_s: float = SEGMENT_SECONDS,
    min_remainder_s: float = MIN_REMAINDER_SECONDS,
) -> Points:
    """Segment ends of the recorded passes with position <= ``last_position``."""
    columns: dict[str, list] = {
        key: []
        for key in (
            "position",
            "segment",
            "e",
            "duration",
            "t",
            "tau",
            "completed",
            "past",
            "force",
            "id",
        )
    }
    past: list[float] = []
    for slot in timeline:
        if slot.position > last_position:
            break
        if not slot.recorded:
            continue
        mean_past = float(np.mean(past)) if past else float("nan")
        for index, end in enumerate(segment_ends(slot.duration_s, segment_s, min_remainder_s), 1):
            columns["position"].append(slot.position)
            columns["segment"].append(index)
            columns["e"].append(end)
            columns["duration"].append(slot.duration_s)
            columns["t"].append(slot.position - 1 + end / slot.duration_s)
            columns["tau"].append(slot.tau_start_s + end)
            columns["completed"].append(slot.position - 1)
            columns["past"].append(mean_past)
            columns["force"].append(slot.force_present)
            columns["id"].append(slot.record_id)
        past.append(slot.duration_s)
    tool = timeline[0].tool if timeline else ""
    return Points(
        tool=tool,
        position=np.asarray(columns["position"], dtype=np.int64),
        segment=np.asarray(columns["segment"], dtype=np.int64),
        elapsed_in_pass=np.asarray(columns["e"], dtype=np.float64),
        duration=np.asarray(columns["duration"], dtype=np.float64),
        t_label=np.asarray(columns["t"], dtype=np.float64),
        tau=np.asarray(columns["tau"], dtype=np.float64),
        completed_passes=np.asarray(columns["completed"], dtype=np.float64),
        mean_past_duration=np.asarray(columns["past"], dtype=np.float64),
        force_present=np.asarray(columns["force"], dtype=bool),
        record_id=tuple(columns["id"]),
    )


@dataclass(frozen=True)
class Targets:
    rul_pass: np.ndarray
    rul_sec: np.ndarray
    y_pass: np.ndarray  # min(RUL, k)
    y_sec: np.ndarray  # min(RUL_sec, k_sec)
    window_pass: np.ndarray  # RUL <= k
    window_sec: np.ndarray  # RUL_sec <= k_sec


def rul_targets(
    points: Points,
    ttf_position: int,
    tau_ttf: float,
    k: float = K_PASSES,
    k_sec: float = K_SECONDS,
) -> Targets:
    rul_pass = ttf_position - points.t_label
    rul_sec = tau_ttf - points.tau
    if (rul_pass < -1e-9).any() or (rul_sec < -1e-6).any():
        raise ValueError("prediction points must not lie after the TTF")
    rul_pass = np.maximum(rul_pass, 0.0)
    rul_sec = np.maximum(rul_sec, 0.0)
    return Targets(
        rul_pass=rul_pass,
        rul_sec=rul_sec,
        y_pass=np.minimum(rul_pass, k),
        y_sec=np.minimum(rul_sec, k_sec),
        window_pass=rul_pass <= k + 1e-9,
        window_sec=rul_sec <= k_sec + 1e-6,
    )


@dataclass(frozen=True)
class ToolLabels:
    """Everything label-related for one tool under one TTF rule and one axis."""

    tool: str
    rule: str
    ttf_record: str
    ttf_position: int
    tau_ttf: float
    wear_at_ttf_um: float  # isotonic-smoothed flank wear of the TTF pass
    points: Points
    targets: Targets


def tool_labels(
    timeline: Sequence[Slot],
    metas: Sequence[PassMeta],
    rule: str = "confirmed",
    k: float = K_PASSES,
    k_sec: float = K_SECONDS,
) -> ToolLabels:
    """TTF, points and targets of one tool. ``metas`` are its recorded passes on the timeline."""
    on_timeline = {slot.record_id for slot in timeline if slot.recorded}
    recorded = sorted(
        (meta for meta in metas if meta.record_id in on_timeline),
        key=lambda meta: (meta.run, meta.block),
    )
    index = ttf_index(rule, recorded)
    if index is None:
        raise ValueError(f"{recorded[0].tool}: no TTF under rule {rule!r} (censored)")
    ttf_record = recorded[index].record_id
    slot = next(slot for slot in timeline if slot.record_id == ttf_record)
    smoothed = isotonic([meta.flank_wear for meta in recorded])
    points = prediction_points(timeline, slot.position)
    return ToolLabels(
        tool=slot.tool,
        rule=rule,
        ttf_record=ttf_record,
        ttf_position=slot.position,
        tau_ttf=slot.tau_end_s,
        wear_at_ttf_um=float(smoothed[index]),
        points=points,
        targets=rul_targets(points, slot.position, slot.tau_end_s, k, k_sec),
    )
