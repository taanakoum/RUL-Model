"""The Nonastreda adapter: raw files -> the generic :class:`rulfuse.core.schema.Dataset`.

Every option is a sensitivity axis of PROJECT_PLAN.md §12; the defaults are the plan's:
causal trim, gap-aware axis, duplicates attached to the latest member, T1's leading run not
counted, the confirmed TTF rule.

Mapping onto the schema (§6.11):

* unit = tool; recording = a recorded pass on the timeline (unrecorded gap passes only
  shift the operating time τ); the signal is the trimmed force of the pass, or None when a
  duplicated recording was attached to another member (D8);
* images = the tool, chip and workpiece photos of each pass, available after its end (D9);
* HI = isotonic-smoothed flank wear (µm); stage = ``tool_label``; group = the run;
* failure time = the end of the TTF pass; passes, runs, blocks and labels go to ``attrs``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from rulfuse import paths
from rulfuse.adapters.nonastreda.failure import (
    ToolFailure,
    isotonic,
    isotonic_causal,
    recorded_on_timeline,
    tool_failure,
)
from rulfuse.adapters.nonastreda.force_store import ForceStore, sha256_file
from rulfuse.adapters.nonastreda.images import PHOTO_KINDS, PHOTO_SPECS
from rulfuse.adapters.nonastreda.load import (
    FORCE_CHANNELS,
    SAMPLE_RATE_HZ,
    PassMeta,
    load_labels,
    metas_by_tool,
)
from rulfuse.adapters.nonastreda.timeline import Slot, build_timelines
from rulfuse.adapters.nonastreda.trim import trim_bounds
from rulfuse.core.schema import (
    ChannelSpec,
    Dataset,
    Image,
    Recording,
    Settings,
    SignalRef,
    Unit,
)

ADAPTER_VERSION = 1
NAME = "nonastreda"
HORIZON_S = 1500.0  # D3
BASELINE_RECORDINGS = 4  # B
SEGMENT_S = 10.0  # D7
CHANNELS = tuple(
    ChannelSpec(name, "force", name[-1], "raw", "N") for name in FORCE_CHANNELS
)  # Fx, Fy, Fz


@dataclass(frozen=True)
class NonastredaOptions:
    trim_rule: str = "causal"
    axis: str = "gap"
    attach: str = "latest"
    count_leading: bool = False
    ttf_rule: str = "confirmed"


@dataclass
class NonastredaData:
    """The adapter's own view, before conversion to the schema (EDA and audits use it)."""

    options: NonastredaOptions
    metas: list[PassMeta]
    store: ForceStore
    bounds: dict[str, tuple[int, int]]  # trimmed (start, end) sample of every record
    durations: dict[str, float]  # trimmed seconds of every record
    timelines: dict[str, list[Slot]]
    failures: dict[str, ToolFailure] = field(default_factory=dict)
    raw: Path = field(default_factory=paths.raw_dir)

    @property
    def tools(self) -> list[str]:
        return list(self.timelines)


@dataclass(frozen=True)
class ForceSignals:
    """The force store as a :class:`SignalSource` (sources are representative record ids)."""

    store: ForceStore

    def read(self, source: str):
        return self.store.force(source)


def trimmed_bounds(store: ForceStore, rule: str = "causal") -> dict[str, tuple[int, int]]:
    """Trim bounds of every record (computed once per distinct recording)."""
    by_representative = {rep: trim_bounds(store.force(rep), rule) for rep in store.distinct_ids()}
    return {record_id: by_representative[rep] for record_id, rep in store.representative.items()}


def load_data(
    options: NonastredaOptions | None = None,
    raw: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> NonastredaData:
    opts = options or NonastredaOptions()
    raw_path = Path(raw) if raw is not None else paths.raw_dir()
    metas = load_labels(raw_path)
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
    failures = {
        tool: tool_failure(timeline, by_tool[tool], opts.ttf_rule)
        for tool, timeline in timelines.items()
    }
    return NonastredaData(opts, metas, store, bounds, durations, timelines, failures, raw_path)


def to_units(data: NonastredaData) -> tuple[Unit, ...]:
    by_tool = metas_by_tool(data.metas)
    group_of = {m: i for i, group in enumerate(data.store.groups) for m in group}
    units = []
    for tool, timeline in data.timelines.items():
        recorded_metas = recorded_on_timeline(timeline, by_tool[tool])
        wear = [meta.flank_wear for meta in recorded_metas]
        smoothed = dict(zip([m.record_id for m in recorded_metas], isotonic(wear), strict=True))
        causal = dict(
            zip([m.record_id for m in recorded_metas], isotonic_causal(wear), strict=True)
        )
        meta_of = {meta.record_id: meta for meta in recorded_metas}
        recordings, images = [], []
        for slot in timeline:
            if not slot.recorded:
                continue
            rid = str(slot.record_id)
            meta = meta_of[rid]
            signal = None
            if slot.force_record is not None:
                start, end = data.bounds[slot.force_record]
                signal = SignalRef(data.store.representative[slot.force_record], start, end)
            recordings.append(
                Recording(
                    key=rid,
                    start_s=slot.tau_start_s,
                    duration_s=slot.duration_s,
                    signal=signal,
                    hi=float(smoothed[rid]),
                    hi_causal=float(causal[rid]),
                    stage=meta.tool_label,
                    group=f"R{slot.run}",
                    attrs={
                        "position": slot.position,
                        "run": slot.run,
                        "block": slot.block,
                        "image_label": meta.image_label,
                        "flank_wear_um": meta.flank_wear,
                        "duplicate_group": group_of.get(rid, -1),
                    },
                )
            )
            for kind, ext in PHOTO_KINDS.items():
                images.append(Image(rid, kind, f"{kind}/{rid}{ext}", slot.tau_end_s))
        failure = data.failures[tool]
        units.append(
            Unit(
                name=tool,
                recordings=tuple(recordings),
                images=tuple(images),
                failure_s=failure.tau_ttf,
                attrs={
                    "ttf_record": failure.ttf_record,
                    "ttf_position": failure.ttf_position,
                    "wear_at_ttf_um": failure.wear_at_ttf_um,
                    "last_position": timeline[-1].position,
                    "unrecorded_passes": sum(not slot.recorded for slot in timeline),
                },
            )
        )
    return tuple(units)


def provenance(data: NonastredaData) -> dict:
    return {
        "adapter_version": ADAPTER_VERSION,
        "force_store": {
            "store_version": data.store.meta["store_version"],
            "source_sha256": data.store.meta["source_sha256"],
            "distinct_recordings": data.store.meta["distinct_recordings"],
        },
        "labels_sha256": {
            name: sha256_file(data.raw / name) for name in ("labels.csv", "labels_reg.csv")
        },
        "options": asdict(data.options),
    }


def to_dataset(data: NonastredaData, segment_s: float = SEGMENT_S) -> Dataset:
    """The schema view. ``segment_s`` is the segment length S (D7: 10 s; a remainder under
    S/2 joins the previous segment). It changes only where the core cuts segments: failure
    times, the HI and the images are the same for every S."""
    if not segment_s > 0:
        raise ValueError("segment_s must be > 0")
    return Dataset(
        name=NAME,
        units=to_units(data),
        channels=CHANNELS,
        settings=Settings(
            horizon_s=HORIZON_S,
            sample_rate_hz=float(SAMPLE_RATE_HZ),
            model_rate_hz=float(SAMPLE_RATE_HZ),
            baseline_recordings=BASELINE_RECORDINGS,
            segment_s=float(segment_s),
            min_remainder_s=float(segment_s) / 2.0,
            hi_name="flank wear (um)",
        ),
        signals=ForceSignals(data.store),
        image_specs=dict(PHOTO_SPECS),
        image_root=data.raw,
        provenance=provenance(data),
    )


def load(
    options: NonastredaOptions | None = None,
    raw: str | Path | None = None,
    store_dir: str | Path | None = None,
    segment_s: float = SEGMENT_S,
) -> Dataset:
    """The Nonastreda dataset in the generic schema."""
    return to_dataset(load_data(options, raw, store_dir), segment_s)
