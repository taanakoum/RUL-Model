"""Generic leakage checks (PROJECT_PLAN.md §6.2 step 4, §15 P9).

``stage_steps``: does a statistic encode a *stage label* rather than the health it should
follow? Rows (images, recordings) are grouped by acquisition group (here a run). A
statistic that follows degradation rises over many group boundaries; one tied to the stage
label (for example an acquisition change made when the stage changed) jumps at a stage
boundary and nowhere else. For every statistic it counts the units whose largest
group-to-group step falls on a stage boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

LEAK_MAX_UNITS = 3  # a statistic leaks if a stage boundary is its largest step in > 3 units


def _transitions(stages: Sequence[int]) -> list[tuple[int, int]]:
    return sorted({(a, b) for a, b in zip(stages, stages[1:], strict=False) if b > a})


def stage_steps(
    units: Sequence[str] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    stages: Sequence[int] | np.ndarray,
    values: Mapping[str, np.ndarray],
) -> dict[str, dict]:
    """Rows must be in time order within each unit. ``values[name]`` holds one value per row
    (NaN rows are skipped). A group's stage is the largest stage among its rows."""
    units, groups, stages = np.asarray(units), np.asarray(groups), np.asarray(stages)
    out: dict[str, dict] = {}
    for name, column in values.items():
        column = np.asarray(column, dtype=np.float64)
        at_stage = 0
        per_transition: dict[str, int] = {}
        shares: dict[str, list[float]] = {}
        used = 0
        for unit in dict.fromkeys(units.tolist()):
            rows = np.flatnonzero(units == unit)
            order: list[str] = list(dict.fromkeys(groups[rows].tolist()))
            means, group_stage = {}, {}
            for group in order:
                in_group = rows[groups[rows] == group]
                finite = column[in_group][np.isfinite(column[in_group])]
                group_stage[group] = int(stages[in_group].max())
                if finite.size:
                    means[group] = float(finite.mean())
            ordered = [g for g in order if g in means]
            if len(ordered) < 3:
                continue
            used += 1
            steps = {b: means[b] - means[a] for a, b in zip(ordered, ordered[1:], strict=False)}
            span = max(means.values()) - min(means.values())
            biggest = max(steps, key=lambda g: abs(steps[g]))
            boundaries = {
                b: (group_stage[a], group_stage[b])
                for a, b in zip(ordered, ordered[1:], strict=False)
                if group_stage[b] > group_stage[a]
            }
            if biggest in boundaries:
                at_stage += 1
            for b, (lo, hi) in boundaries.items():
                label = f"{lo}to{hi}"
                per_transition[label] = per_transition.get(label, 0) + int(biggest == b)
                shares.setdefault(label, []).append(abs(steps[b]) / span if span > 0 else 0.0)
        out[name] = {
            "units": used,
            "largest_at_stage_boundary": at_stage,
            "largest_by_transition": per_transition,
            "median_share": {k: round(float(np.median(v)), 2) for k, v in shares.items()},
            "leaks": at_stage > LEAK_MAX_UNITS,
        }
    return out


def mean_luminance(path: str | Path) -> float:
    """Mean grey level (0-1) of an image, read at reduced size."""
    from PIL import Image

    with Image.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", (image.width // 4, image.height // 4))
        grey = image.convert("L")
    grey.thumbnail((256, 256))
    return float(np.asarray(grey, dtype=np.float64).mean() / 255.0)
