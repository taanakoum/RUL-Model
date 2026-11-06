"""The arms of the modality study and its analysis (docs/OMNIFUSER_RUL_PLAN.md §8).

Players: ``force`` and the admitted image types. Usage is on in every arm. The arms:

* every non-empty subset of the players, and the usage-only arm (the empty coalition);
* ``force`` and the first image type with usage off (sensing without usage, Q4);
* all players trained with modality dropout 0.2 (reliance, Q3);
* all players without the memory, and without the temporal loss terms (the two ablations of
  version 3: do the memory (O8) and the monotonic and slope terms (O9) help?).

Shapley values and the paired tests use only the arms with usage on, no dropout and the
base memory and loss settings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np

from rulfuse.core.store import SegmentStore
from rulfuse.omnifuser_rul import aggregate
from rulfuse.omnifuser_rul.train import FORCE, ArmSpec, arm_dir


def modality_arms(players: Sequence[str], base: ArmSpec | None = None) -> list[ArmSpec]:
    base = base or ArmSpec(players=())
    players = tuple(players)
    arms = [replace(base, players=())]
    for size in range(1, len(players) + 1):
        arms += [replace(base, players=tuple(c)) for c in combinations(players, size)]
    photos = [p for p in players if p != FORCE]
    if FORCE in players:
        arms.append(replace(base, players=(FORCE,), usage=False))
    if photos:
        arms.append(replace(base, players=(photos[0],), usage=False))
    arms.append(replace(base, players=players, modality_dropout=0.2))
    if base.memory:
        arms.append(replace(base, players=players, memory=False))
    if base.temporal:
        arms.append(replace(base, players=players, mono_weight=0.0, slope_weight=0.0))
    return arms


def is_coalition_arm(spec: ArmSpec, base: ArmSpec) -> bool:
    """An arm of the modality game: usage on, no dropout, the base memory and loss."""
    return (
        spec.usage
        and not spec.modality_dropout
        and spec.memory == base.memory
        and spec.mono_weight == base.mono_weight
        and spec.slope_weight == base.slope_weight
    )


def analyse(
    store: SegmentStore,
    root: Path,
    players: Sequence[str],
    base: ArmSpec | None = None,
    after_freeze: bool = False,
    today: date | None = None,
) -> dict:
    """Validation scores of every arm; with ``after_freeze``, test scores, Shapley values
    and the two-versus-one tests (choices made on the validation units)."""
    base = base or ArmSpec(players=())
    arms = modality_arms(players, base)
    validation = {}
    for spec in arms:
        folder = arm_dir(root, store, spec)
        if folder.is_dir():
            validation[spec.name] = aggregate.score_arm(store, folder, "validation")
    report: dict = {"validation": validation, "b2_validation": aggregate.b2_by_unit(store)}
    if not after_freeze:
        return report
    aggregate.check_seal("test", after_freeze, today)
    test = {
        spec.name: aggregate.score_arm(store, arm_dir(root, store, spec), "test", True, today)
        for spec in arms
    }
    report["test"] = test

    def per_unit(name: str, role: dict) -> dict[str, float]:
        return {u: s["window_mae"] for u, s in role[name]["per_unit"].items() if "window_mae" in s}

    coalitions = {
        frozenset(spec.players): spec.name
        for spec in arms
        if is_coalition_arm(spec, base) and spec.players
    }
    window = {c: per_unit(name, test) for c, name in coalitions.items()}
    validation_mean = {
        c: validation[name]["summary"]["window_mae"]["mean"] for c, name in coalitions.items()
    }
    usage_only = replace(base, players=()).name
    b2_validation = float(np.mean(list(report["b2_validation"].values())))
    empty_is_b2 = b2_validation <= validation[usage_only]["summary"]["window_mae"]["mean"]
    empty = aggregate.b2_by_unit(store, "test") if empty_is_b2 else per_unit(usage_only, test)
    report["empty_coalition"] = "B2" if empty_is_b2 else usage_only
    report["modality"] = aggregate.modality_report(window, validation_mean, list(players), empty)
    return report
