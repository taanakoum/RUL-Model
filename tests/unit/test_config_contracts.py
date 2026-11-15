"""Serialization and approved numeric defaults are part of the reproducibility contract."""

from __future__ import annotations

import json
from dataclasses import fields

import pytest

from rulfuse.config import FitConfig
from rulfuse.train import fit_key

pytestmark = [pytest.mark.unit, pytest.mark.requirements("P14", "P20", "P22")]


def test_default_schedule_and_regularization_match_plan():
    config = FitConfig()
    assert (config.phase_a_steps, config.max_steps, config.eval_every, config.patience) == (
        100,
        500,
        10,
        10,
    )
    assert (config.warmup_steps, config.nll_ramp_steps, config.ema_decay) == (20, 50, 0.99)
    assert (config.d_model, config.rank, config.layers, config.heads, config.ffn) == (
        64,
        16,
        2,
        4,
        128,
    )
    assert (config.modality_dropout, config.all_modality_dropout) == (0.3, 0.05)
    assert (config.channel_dropout, config.token_dropout, config.feature_noise) == (0.1, 0.1, 0.1)
    assert config.mono_weight == pytest.approx(1 / 15)


def test_json_roundtrip_preserves_every_fit_setting_and_identity(tmp_path):
    config = FitConfig(players=("signal", "edge"), fusion="F0", extras={"description": "test"})
    config.save(tmp_path / "config.json")
    restored = FitConfig.from_dict(json.loads((tmp_path / "config.json").read_text()))
    assert restored == config
    assert fit_key(restored) == fit_key(config)
    assert set(config.to_dict()) == {field.name for field in fields(config)}


@pytest.mark.parametrize(
    "change",
    [
        {"fusion": "F0"},
        {"players": ["signal", "tool"]},
        {"usage": False},
        {"token_mode": "absolute"},
        {"memory": False},
        {"lr": 0.003},
        {"mono_weight": 0.0},
    ],
)
def test_material_fit_changes_cannot_share_an_identity(change):
    config = FitConfig()
    assert fit_key(config) != fit_key(config.with_overrides(change))
