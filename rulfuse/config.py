"""The settings of one RUL-Fuse fit (PROJECT_PLAN.md Appendix A, §6, §7).

Every default here is fixed before any run (§7.5). A study arm is the defaults plus the
decisions inherited from earlier gates plus the arm's own overrides
(:func:`FitConfig.with_overrides`); the overrides, not the full config, name an arm.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

FUSIONS = ("F1", "F0")
TOKEN_MODES = ("both", "absolute", "relative")


@dataclass(frozen=True)
class FitConfig:
    # inputs
    players: tuple[str, ...] = ("signal",)  # "signal" and image kinds, e.g. ("signal", "tool")
    usage: bool = True  # usage features U (operating time, seconds into the recording)
    token_mode: str = "both"  # §5.9
    history_recordings: int = 0  # 0: the whole history; k: only the last k recordings (controls)
    oracle_hi: bool = False  # sanity check 4: the measured causal HI as an input
    # model (§6.3, §6.4, §6.6)
    fusion: str = "F1"
    d_model: int = 64
    rank: int = 16
    layers: int = 2
    heads: int = 4
    ffn: int = 128
    dropout: float = 0.2
    memory: bool = True  # D21
    image_hidden: int = 32  # F0's image GRU
    time_buckets: int = 32  # relative-time bias resolution
    # loss (§6.7, D22)
    nll_weight: float = 0.5
    hi_weight: float = 0.3
    cons_weight: float = 0.05
    mono_weight: float = 1.0 / 15.0
    pair_max_fraction: float = 0.1333  # pairs at most 0.13·H apart
    w_far: float = 0.3
    expert_weight: float = 0.5  # F0 deep supervision
    # augmentation (§6.8)
    trajectory_aug: bool = True
    rho_max: float = 0.2
    modality_dropout: float = 0.3
    all_modality_dropout: float = 0.05
    channel_dropout: float = 0.1
    token_dropout: float = 0.1
    feature_noise: float = 0.1
    copies: int = 4
    # optimisation and schedule (§7.3, §7.4)
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    phase_a: bool = True
    phase_a_steps: int = 100
    warmup_steps: int = 20
    max_steps: int = 500
    nll_ramp_steps: int = 50
    eval_every: int = 10
    patience: int = 10
    ema_decay: float = 0.99
    min_lr: float = 1e-5
    regularize: bool = True  # sanity check 1 switches dropout, noise and augmentation off
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        if self.token_mode not in TOKEN_MODES:
            raise ValueError(f"token_mode must be one of {TOKEN_MODES}")
        object.__setattr__(self, "players", tuple(self.players))
        self._check_values()

    def _check_values(self) -> None:
        """Reject settings no fit can use, before any training (test review T07)."""
        problems = []

        def finite(name: str) -> float:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                problems.append(f"{name} must be finite")
            return value

        for name in (
            "dropout",
            "modality_dropout",
            "all_modality_dropout",
            "channel_dropout",
            "token_dropout",
            "ema_decay",
        ):
            if not 0.0 <= finite(name) <= 1.0:
                problems.append(f"{name} must be in [0, 1]")
        for name in (
            "rho_max",
            "feature_noise",
            "weight_decay",
            "min_lr",
            "w_far",
            "nll_weight",
            "hi_weight",
            "cons_weight",
            "mono_weight",
            "expert_weight",
        ):
            if finite(name) < 0:
                problems.append(f"{name} must be >= 0")
        for name in ("lr", "grad_clip", "pair_max_fraction"):
            if not finite(name) > 0:
                problems.append(f"{name} must be > 0")
        for name in ("d_model", "rank", "layers", "heads", "ffn", "copies", "eval_every"):
            if int(getattr(self, name)) < 1:
                problems.append(f"{name} must be >= 1")
        for name in ("phase_a_steps", "warmup_steps", "nll_ramp_steps", "patience"):
            if int(getattr(self, name)) < 0:
                problems.append(f"{name} must be >= 0")
        if self.heads >= 1 and self.d_model % self.heads:
            problems.append("d_model must be a multiple of heads")
        if self.max_steps <= self.phase_a_steps:
            problems.append("max_steps must exceed phase_a_steps (phase B needs steps)")
        if len(set(self.players)) != len(self.players):
            problems.append("players must not repeat")
        if problems:
            raise ValueError("invalid fit configuration: " + "; ".join(problems))

    @property
    def image_kinds(self) -> tuple[str, ...]:
        return tuple(p for p in self.players if p != "signal")

    @property
    def uses_signal(self) -> bool:
        return "signal" in self.players

    def with_overrides(self, overrides: dict | None) -> FitConfig:
        if not overrides:
            return self
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        values = dict(overrides)
        if "players" in values:
            values["players"] = tuple(values["players"])
        return replace(self, **values)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["players"] = list(self.players)
        return out

    @classmethod
    def from_dict(cls, values: dict) -> FitConfig:
        return cls().with_overrides(values)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
