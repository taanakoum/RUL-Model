"""Nonastreda failure time and health indicator (PROJECT_PLAN.md D2, §2.3, §5.6).

TTF rules, applied to a tool's *recorded* label sequence in (run, block) order:

* ``confirmed`` (D2, default): the first ``dulled`` pass followed by at least 2 more
  ``dulled`` passes among its next 3 recorded passes;
* ``literal``: the first ``dulled`` pass;
* ``wear110``: the first pass whose isotonic-smoothed flank wear reaches 110 µm;
* ``tool_label3``: the first pass with ``tool_label`` 3;
* ``dulled_onwards``: the first pass after which every pass is ``dulled``.

The failure time τ* is the end of the TTF pass, in seconds of cutting on the chosen axis.
The health indicator is the isotonic-smoothed flank wear (full: a target; causal: an
oracle input).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import isotonic_regression

from rulfuse.adapters.nonastreda.load import PassMeta
from rulfuse.adapters.nonastreda.timeline import Slot

TTF_RULES = ("confirmed", "literal", "wear110", "tool_label3", "dulled_onwards")
WEAR_THRESHOLD_UM = 110.0
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
class ToolFailure:
    tool: str
    rule: str
    ttf_record: str
    ttf_position: int  # position on the axis (reporting only)
    tau_ttf: float  # failure time τ*: the end of the TTF pass
    wear_at_ttf_um: float  # isotonic-smoothed flank wear of the TTF pass


def recorded_on_timeline(timeline: Sequence[Slot], metas: Sequence[PassMeta]) -> list[PassMeta]:
    on_timeline = {slot.record_id for slot in timeline if slot.recorded}
    return sorted(
        (meta for meta in metas if meta.record_id in on_timeline),
        key=lambda meta: (meta.run, meta.block),
    )


def tool_failure(
    timeline: Sequence[Slot], metas: Sequence[PassMeta], rule: str = "confirmed"
) -> ToolFailure:
    """The TTF of one tool. ``metas`` are its recorded passes."""
    recorded = recorded_on_timeline(timeline, metas)
    index = ttf_index(rule, recorded)
    if index is None:
        raise ValueError(f"{recorded[0].tool}: no TTF under rule {rule!r} (censored)")
    ttf_record = recorded[index].record_id
    slot = next(slot for slot in timeline if slot.record_id == ttf_record)
    smoothed = isotonic([meta.flank_wear for meta in recorded])
    return ToolFailure(
        tool=slot.tool,
        rule=rule,
        ttf_record=ttf_record,
        ttf_position=slot.position,
        tau_ttf=slot.tau_end_s,
        wear_at_ttf_um=float(smoothed[index]),
    )
