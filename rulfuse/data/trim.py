"""Pass trimming: remove the entry and exit transients of a pass (PROJECT_PLAN.md §5.2).

``span_bounds_omnifuser`` is vendored unchanged from OmniFuser
(``omnifuser/data.py::_nonastreda_span_bounds``, commit e3b9985):

1. the force magnitude ``|F|^2`` per sample, smoothed with a 20-sample moving mean, square
   root: the envelope;
2. the reference level: the median envelope over the first 5 % of the pass;
3. active samples: envelope >= 50 % of the reference; keep the first to the last active one.

Step 2 needs the length of the pass, which is unknown while the pass is running, so the
default here, ``span_bounds_causal``, takes the reference over the first 5 s (5,000 samples,
about 5 % of a median 97 s pass). Everything else is unchanged (D14). The centred
20-sample envelope looks 10 ms ahead, which is negligible against 10 s segments. The end
bound is the last active sample, known when the pass ends, which is also when the last
segment's prediction is made.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import uniform_filter1d

PASS_TRIM_WINDOW_SAMPLES = 20
PASS_TRIM_REFERENCE_FRACTION = 0.05
PASS_TRIM_THRESHOLD_FRACTION = 0.5
CAUSAL_REFERENCE_SAMPLES = 5000


def envelope(force: np.ndarray, window: int = PASS_TRIM_WINDOW_SAMPLES) -> np.ndarray:
    magnitude = np.square(np.asarray(force, dtype=np.float64)).sum(axis=1)
    return np.sqrt(uniform_filter1d(magnitude, size=window, mode="nearest"))


def _bounds(force: np.ndarray, reference_length: int) -> tuple[int, int]:
    window = PASS_TRIM_WINDOW_SAMPLES
    if len(force) < window:
        return 0, len(force)
    env = envelope(force, window)
    reference_level = float(np.median(env[: max(window, reference_length)]))
    if not np.isfinite(reference_level) or reference_level <= np.finfo(float).eps:
        return 0, len(force)
    active = np.flatnonzero(env >= reference_level * PASS_TRIM_THRESHOLD_FRACTION)
    if active.size == 0:
        return 0, len(force)
    return int(active[0]), int(active[-1]) + 1


def span_bounds_omnifuser(force: np.ndarray) -> tuple[int, int]:
    """OmniFuser's rule: the reference is the first 5 % of the pass (not causal)."""
    return _bounds(force, int(len(force) * PASS_TRIM_REFERENCE_FRACTION))


def span_bounds_causal(
    force: np.ndarray, reference_samples: int = CAUSAL_REFERENCE_SAMPLES
) -> tuple[int, int]:
    """The causal rule: the reference is the first ``reference_samples`` samples."""
    return _bounds(force, min(reference_samples, len(force)))


TRIM_RULES: dict[str, Callable[[np.ndarray], tuple[int, int]]] = {
    "causal": span_bounds_causal,
    "omnifuser": span_bounds_omnifuser,
}


def trim_bounds(force: np.ndarray, rule: str = "causal") -> tuple[int, int]:
    if rule not in TRIM_RULES:
        raise ValueError(f"trim rule must be one of {sorted(TRIM_RULES)}")
    return TRIM_RULES[rule](force)
