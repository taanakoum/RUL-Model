"""Force channels (PROJECT_PLAN.md D4, §5.4).

The three measured channels are always used; two derived channels are added:

* ``Fz_rel``: Fz minus its mean over the first second of the (trimmed) pass. This removes
  the drift's sawtooth level (OmniFuser F3) and keeps its dynamics. It is causal: the
  offset is known once the pass's first second has passed, before the first segment ends.
* ``F``: the resultant force sqrt(Fx² + Fy² + Fz²).
"""

from __future__ import annotations

import numpy as np

from rulfuse.data.nonastreda import SAMPLE_RATE_HZ

CHANNELS = ("Fx", "Fy", "Fz", "Fz_rel", "F")
FZ_OFFSET_SECONDS = 1.0


def fz_offset(trimmed_pass: np.ndarray, seconds: float = FZ_OFFSET_SECONDS) -> float:
    """Mean Fz over the first ``seconds`` of a trimmed pass."""
    head = np.asarray(trimmed_pass)[: max(1, int(seconds * SAMPLE_RATE_HZ)), 2]
    return float(np.mean(head, dtype=np.float64))


def derived_channels(force: np.ndarray, offset: float) -> np.ndarray:
    """(samples, 3) Fx, Fy, Fz -> (samples, 5) Fx, Fy, Fz, Fz_rel, F (float32)."""
    force = np.asarray(force, dtype=np.float32)
    out = np.empty((len(force), len(CHANNELS)), dtype=np.float32)
    out[:, :3] = force
    out[:, 3] = force[:, 2] - np.float32(offset)
    out[:, 4] = np.sqrt(np.square(force, dtype=np.float32).sum(axis=1))
    return out
