"""P22 (part): the adapter contract is enforced (PROJECT_PLAN.md §6.11)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rulfuse.core.schema import (
    ArraySignals,
    ChannelSpec,
    Dataset,
    Image,
    ImageSpec,
    Recording,
    SchemaError,
    Settings,
    SignalRef,
    Unit,
    validate,
)


def _tiny(**changes) -> Dataset:
    rec = (
        Recording("a1", 0.0, 2.0, SignalRef("s", 0, 2000)),
        Recording("a2", 2.0, 2.0, SignalRef("s", 2000, 4000)),
    )
    unit = Unit("A", rec, (Image("a1", "cam", "a1.png", 2.0),), failure_s=4.0)
    base = Dataset(
        "tiny",
        (unit,),
        (ChannelSpec("x", "acceleration", "x"),),
        Settings(horizon_s=3.0, sample_rate_hz=1000.0),
        ArraySignals({"s": np.zeros((4000, 1), np.float32)}),
        {"cam": ImageSpec(8, 8)},
    )
    return replace(base, **changes)


def test_a_valid_dataset_passes():
    validate(_tiny(), check_signals=True)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: replace(d, settings=replace(d.settings, horizon_s=0.0)), "horizon"),
        (lambda d: replace(d, channels=(ChannelSpec("x", "a", "w"),)), "axis"),
        (
            lambda d: replace(
                d,
                units=(
                    replace(
                        d.units[0],
                        recordings=(
                            d.units[0].recordings[0],
                            replace(d.units[0].recordings[1], start_s=1.0),
                        ),
                    ),
                ),
            ),
            "starts before",
        ),
        (lambda d: replace(d, units=(replace(d.units[0], failure_s=9.0),)), "failure"),
        (lambda d: replace(d, image_specs={}), "ImageSpec"),
        (
            lambda d: replace(
                d,
                units=(
                    replace(
                        d.units[0],
                        recordings=(
                            replace(d.units[0].recordings[0], duration_s=3.0),
                            d.units[0].recordings[1],
                        ),
                    ),
                ),
            ),
            "duration",
        ),
        (lambda d: replace(d, units=(d.units[0], d.units[0])), "duplicate"),
    ],
)
def test_broken_rules_are_reported(change, message):
    with pytest.raises(SchemaError, match=message):
        validate(change(_tiny()))


def test_signal_check_reads_the_source():
    short = _tiny(signals=ArraySignals({"s": np.zeros((100, 1), np.float32)}))
    with pytest.raises(SchemaError, match="samples"):
        validate(short, check_signals=True)
    nan = np.zeros((4000, 1), np.float32)
    nan[5] = math.nan
    with pytest.raises(SchemaError, match="non-finite"):
        validate(_tiny(signals=ArraySignals({"s": nan})), check_signals=True)


@pytest.mark.data
def test_the_nonastreda_adapter_output_validates(dataset):
    validate(dataset, check_signals=True)
    assert len(dataset.units) == 10
    assert sum(len(u.recordings) for u in dataset.units) == 512
    assert {c.name for c in dataset.channels} == {"Fx", "Fy", "Fz"}
    missing = sum(r.signal is None for u in dataset.units for r in u.recordings)
    assert missing == 134  # the non-latest duplicate members (D8)
