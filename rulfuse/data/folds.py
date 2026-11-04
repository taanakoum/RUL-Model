"""Leave-one-tool-out folds (PROJECT_PLAN.md D1, §4.1).

Vendored from OmniFuser (``omnifuser/data.py``: ``ToolFold``, ``leave_one_tool_out_folds``,
commit e3b9985): fold *i* tests tool T*i*, validates on the next tool cyclically and trains
on the other 8.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolFold:
    train_tools: tuple[str, ...]
    validation_tools: tuple[str, ...]
    test_tools: tuple[str, ...]

    @property
    def test(self) -> str:
        return self.test_tools[0]

    @property
    def validation(self) -> str:
        return self.validation_tools[0]


def leave_one_tool_out_folds(tools: Iterable[str]) -> Iterator[ToolFold]:
    ordered = sorted(set(tools), key=lambda item: int(item[1:]))
    if len(ordered) < 3:
        raise ValueError("leave-one-tool-out with a separate validation tool needs >= 3 tools")
    for index, test_tool in enumerate(ordered):
        validation_tool = ordered[(index + 1) % len(ordered)]
        train = tuple(tool for tool in ordered if tool not in {test_tool, validation_tool})
        yield ToolFold(train, (validation_tool,), (test_tool,))
