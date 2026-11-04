"""Assemble labels, the force store, trims, timelines and RUL labels in one call.

Every option is a sensitivity axis of PROJECT_PLAN.md §12; the defaults are the plan's:
causal trim, gap-aware axis, duplicates attached to the latest member, T1's leading run not
counted, the confirmed TTF rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rulfuse import paths
from rulfuse.data.force_store import ForceStore
from rulfuse.data.labels import K_PASSES, K_SECONDS, ToolLabels, tool_labels
from rulfuse.data.nonastreda import SAMPLE_RATE_HZ, PassMeta, load_labels, metas_by_tool
from rulfuse.data.timeline import Slot, build_timelines
from rulfuse.data.trim import trim_bounds


@dataclass(frozen=True)
class DatasetOptions:
    trim_rule: str = "causal"
    axis: str = "gap"
    attach: str = "latest"
    count_leading: bool = False
    ttf_rule: str = "confirmed"
    k: float = K_PASSES
    k_sec: float = K_SECONDS


@dataclass
class Dataset:
    options: DatasetOptions
    metas: list[PassMeta]
    store: ForceStore
    bounds: dict[str, tuple[int, int]]  # trimmed (start, end) sample of every record
    durations: dict[str, float]  # trimmed seconds of every record
    timelines: dict[str, list[Slot]]
    labels: dict[str, ToolLabels] = field(default_factory=dict)

    @property
    def tools(self) -> list[str]:
        return list(self.timelines)


def trimmed_bounds(store: ForceStore, rule: str = "causal") -> dict[str, tuple[int, int]]:
    """Trim bounds of every record (computed once per distinct recording)."""
    by_representative = {rep: trim_bounds(store.force(rep), rule) for rep in store.distinct_ids()}
    return {record_id: by_representative[rep] for record_id, rep in store.representative.items()}


def load_dataset(
    options: DatasetOptions | None = None,
    raw: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> Dataset:
    opts = options or DatasetOptions()
    metas = load_labels(raw or paths.raw_dir())
    store = ForceStore.open(store_dir or paths.force_store_dir())
    bounds = trimmed_bounds(store, opts.trim_rule)
    durations = {rid: (end - start) / SAMPLE_RATE_HZ for rid, (start, end) in bounds.items()}
    timelines = build_timelines(
        metas,
        durations,
        store.groups,
        axis=opts.axis,
        count_leading=opts.count_leading,
        attach=opts.attach,
    )
    by_tool = metas_by_tool(metas)
    labels = {
        tool: tool_labels(timeline, by_tool[tool], opts.ttf_rule, opts.k, opts.k_sec)
        for tool, timeline in timelines.items()
    }
    return Dataset(opts, metas, store, bounds, durations, timelines, labels)
