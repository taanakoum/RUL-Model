"""Leave-one-unit-out folds (PROJECT_PLAN.md D1, §4.1).

Vendored from OmniFuser (``omnifuser/data.py``: ``leave_one_tool_out_folds``, commit
e3b9985), with units in the order the adapter gives: fold *i* tests unit *i*, validates on
the next unit cyclically and trains on the others.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    test: str
    validation: str
    train: tuple[str, ...]


def leave_one_unit_out(units: Iterable[str]) -> list[Fold]:
    ordered = list(dict.fromkeys(units))
    if len(ordered) < 3:
        raise ValueError("leave-one-unit-out with a separate validation unit needs >= 3 units")
    folds = []
    for index, test in enumerate(ordered):
        validation = ordered[(index + 1) % len(ordered)]
        train = tuple(unit for unit in ordered if unit not in {test, validation})
        folds.append(Fold(test, validation, train))
    return folds
