"""A synthetic toy adapter: the proof that the core transfers (PROJECT_PLAN.md §6.11, P22).

Nothing like Nonastreda: two accelerometer channels at 2 kHz (no 3-axis group, so no
magnitude channel), one image type, H = 600 s, back-to-back recordings (no gaps), no
duplicates, every recording a group of its own. Degradation is a power law in operating
time; vibration amplitude and image brightness follow it, so a working pipeline can learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from rulfuse.core.schema import (
    ArraySignals,
    ChannelSpec,
    Dataset,
    Image,
    ImageSpec,
    Recording,
    Settings,
    SignalRef,
    Unit,
)

NAME = "toy"
RATE_HZ = 2000.0
THRESHOLD = 100.0
CHANNELS = (
    ChannelSpec("acc_x", "acceleration", "x", "raw", "g"),
    ChannelSpec("acc_y", "acceleration", "y", "raw", "g"),
)


def _wear(t: np.ndarray, rate: float) -> np.ndarray:
    return 20.0 + rate * np.power(t, 1.5)


def build(
    root: str | Path,
    n_units: int = 5,
    seed: int = 0,
    horizon_s: float = 600.0,
    recording_s: tuple[float, float] = (20.0, 30.0),
    after_failure: int = 3,
) -> Dataset:
    """Generate the toy dataset; images are written under ``root``."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    units = []
    for u in range(n_units):
        name = f"U{u + 1}"
        life = rng.uniform(800.0, 1100.0)
        rate = (THRESHOLD - 20.0) / life**1.5
        recordings, images, t, failure = [], [], 0.0, None
        index, extra = 0, 0
        while True:
            index += 1
            duration = float(np.round(rng.uniform(*recording_s), 3))
            n = int(round(duration * RATE_HZ))
            time = t + np.arange(n) / RATE_HZ
            level = 1.0 + 0.02 * _wear(time, rate)
            phase = rng.uniform(0, 2 * np.pi)
            x = level * np.sin(2 * np.pi * 120.0 * time + phase) + rng.normal(0, 0.2, n)
            y = 0.5 * level * np.cos(2 * np.pi * 45.0 * time) + 0.001 * (time - t)
            key = f"{name}-r{index:02d}"
            arrays[key] = np.stack([x, y], axis=1).astype(np.float32)
            end = t + n / RATE_HZ
            hi = float(_wear(np.array([end]), rate)[0])
            stage = 1 if hi < 50 else 2 if hi < THRESHOLD else 3
            recordings.append(
                Recording(
                    key=key,
                    start_s=t,
                    duration_s=n / RATE_HZ,
                    signal=SignalRef(key, 0, n),
                    hi=hi,
                    hi_causal=hi,
                    stage=stage,
                    group=key,
                )
            )
            folder = root / name
            folder.mkdir(parents=True, exist_ok=True)
            grey = np.clip(40 + 1.5 * hi + rng.normal(0, 8, (48, 48)), 0, 255).astype(np.uint8)
            PILImage.fromarray(grey, mode="L").save(folder / f"{key}.png")
            images.append(Image(key, "surface", f"{name}/{key}.png", end))
            t = end
            if failure is not None:
                extra += 1
            elif hi >= THRESHOLD:
                failure = end
            if failure is not None and extra >= after_failure:
                break
        units.append(Unit(name, tuple(recordings), tuple(images), failure))
    return Dataset(
        name=NAME,
        units=tuple(units),
        channels=CHANNELS,
        settings=Settings(
            horizon_s=horizon_s,
            sample_rate_hz=RATE_HZ,
            model_rate_hz=1000.0,
            baseline_recordings=3,
            hi_name="synthetic wear",
        ),
        signals=ArraySignals(arrays),
        image_specs={"surface": ImageSpec(32, 32)},
        image_root=root,
        provenance={"adapter": NAME, "seed": seed, "units": n_units, "horizon_s": horizon_s},
    )
