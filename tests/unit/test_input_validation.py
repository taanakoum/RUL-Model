"""Reject nonphysical settings before they can create invalid labels, caches, or fits.

These are strict acceptance tests: missing validation is a failure, not an xfail.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rulfuse.config import FitConfig
from rulfuse.core.schema import (
    ArraySignals, ChannelSpec, Dataset, Image, ImageSpec, Recording, SchemaError, Settings, Unit,
    validate,
)
from rulfuse.core.targets import rul_targets
from rulfuse.uncertainty import kalman_rul, mixture_moments

pytestmark = [pytest.mark.unit, pytest.mark.adversarial,
              pytest.mark.requirements("P5", "P8", "P12", "P15", "P19", "P22")]


def valid_schema() -> Dataset:
    return Dataset(
        "validation", (Unit("A", (Recording("a", 0., 4.),), failure_s=4.),),
        (ChannelSpec("x", "acceleration", "x"),),
        Settings(horizon_s=3., sample_rate_hz=64.), ArraySignals({}),
    )


@pytest.mark.parametrize("field", ["horizon_s", "sample_rate_hz", "model_rate_hz"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_schema_requires_finite_time_and_sampling_settings(field, value):
    ds = valid_schema()
    with pytest.raises(SchemaError):
        validate(replace(ds, settings=replace(ds.settings, **{field: value})))


def test_schema_rejects_infinite_recording_duration():
    ds = valid_schema()
    unit = Unit("A", (Recording("a", 0., float("inf")),), failure_s=None)
    with pytest.raises(SchemaError):
        validate(replace(ds, units=(unit,)))


@pytest.mark.parametrize("spec", [ImageSpec(0, 8), ImageSpec(8, -1),
                                      ImageSpec(8, 8, (.8, 0, .2, 1))])
def test_schema_rejects_unusable_image_preparation(spec):
    ds = valid_schema()
    unit = replace(ds.units[0], images=(Image("a", "cam", "a.png", 4.),))
    with pytest.raises(SchemaError):
        validate(replace(ds, units=(unit,), image_specs={"cam": spec}))


@pytest.mark.parametrize("changes", [
    {"modality_dropout": -0.1}, {"modality_dropout": 1.1}, {"token_dropout": float("nan")},
    {"channel_dropout": 1.1}, {"all_modality_dropout": -0.1}, {"rho_max": -0.1},
    {"lr": 0.0}, {"lr": float("nan")}, {"weight_decay": -1.0}, {"grad_clip": -1.0},
    {"heads": 0}, {"d_model": 10, "heads": 4}, {"copies": 0}, {"eval_every": 0},
    {"max_steps": 100, "phase_a_steps": 100}, {"ema_decay": 1.1},
    {"w_far": -0.1}, {"players": ["signal", "signal"]},
], ids=lambda changes: ",".join(f"{k}={v}" for k, v in changes.items()))
def test_invalid_fit_configuration_is_rejected_before_training(changes):
    with pytest.raises(ValueError):
        FitConfig.from_dict(changes)


@pytest.mark.parametrize("horizon", [-1., 0., float("inf"), float("nan")])
def test_target_generation_refuses_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        rul_targets(np.array([1., 2.]), 3., horizon)


@pytest.mark.parametrize("sigma", [0., -1., float("nan"), float("inf")])
def test_mixture_refuses_invalid_student_t_scale(sigma):
    with pytest.raises(ValueError):
        mixture_moments(np.array([[1.]]), np.array([[sigma]]), np.array([[4.]]))


@pytest.mark.parametrize("q,variance", [(-1., 1.), (1., -1.), (float("nan"), 1.),
                                         (1., float("nan"))])
def test_kalman_refuses_nonphysical_noise(q, variance):
    with pytest.raises(ValueError):
        kalman_rul(np.array([0., 1.]), np.array([5., 4.]), np.full(2, variance), q, 10., 1.)
