"""Unit-level statistics and the gate rules (PROJECT_PLAN.md §10.3, §4.4, §15.1 row P17).

Bootstrap over units (tools), paired Wilcoxon signed-rank, Holm step-down, and the two
adoption rules A and B that every gate applies with a fixed threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from scipy import stats as sp_stats

_EPS = 1e-9  # float slack for "at least the threshold" comparisons


def _resampled(
    values: np.ndarray, statistic: Callable[..., float], draws: int, seed: int
) -> np.ndarray:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    samples = v[rng.integers(0, len(v), size=(draws, len(v)))]
    try:
        return np.asarray(statistic(samples, axis=1), float)
    except TypeError:  # a statistic without an axis argument
        return np.array([statistic(s) for s in samples], float)


def unit_bootstrap(
    values: np.ndarray,
    statistic: Callable[..., float] = np.mean,
    level: float = 0.95,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval of ``statistic`` over units resampled with replacement
    (PROJECT_PLAN.md §10.3). Reproducible by ``seed``; with fewer than 2 units it collapses."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(statistic(v)), float(statistic(v)))
    stats = _resampled(v, statistic, draws, seed)
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[..., float] = np.mean,
    level: float = 0.95,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval of ``statistic`` of (b − a) over paired units."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape: {a.shape} vs {b.shape}")
    return unit_bootstrap(b - a, statistic, level, draws, seed)


def wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    """Wilcoxon signed-rank test on b − a (``alternative`` as in SciPy: "greater" means b > a).
    All-zero differences give p = 1.0 (SciPy would raise)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    if not np.any(d != 0):
        return {"statistic": 0.0, "p": 1.0}
    res = sp_stats.wilcoxon(d, alternative=alternative)
    return {"statistic": float(res.statistic), "p": float(res.pvalue)}


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values: sorted ascending, p_(i) × (m − i + 1),
    made monotone with a running maximum and capped at 1; returned in the input order."""
    p = np.asarray(pvalues, float)
    m = len(p)
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    scaled = p[order] * (m - np.arange(m))
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def rule_a(
    baseline_mae: np.ndarray,
    option_mae: np.ndarray,
    threshold_s: float,
    min_wins: int = 7,
    level: float = 0.95,
    draws: int = 2000,
    seed: int = 0,
) -> dict:
    """Rule A (§4.4): adopt the option iff the mean per-unit MAE gain (baseline − option) is at
    least ``threshold_s`` AND the option wins (strictly lower MAE) on >= ``min_wins`` units."""
    base, opt = np.asarray(baseline_mae, float), np.asarray(option_mae, float)
    if base.shape != opt.shape:
        raise ValueError(f"per-unit arrays differ in shape: {base.shape} vs {opt.shape}")
    gain = float(np.mean(base - opt))
    wins = int(np.sum(opt < base))
    lo, hi = paired_bootstrap(opt, base, level=level, draws=draws, seed=seed)
    return {
        "gain_mean": gain,
        "wins": wins,
        "n": int(len(base)),
        "adopted": bool(gain >= threshold_s - _EPS and wins >= min_wins),
        "gain_ci": (lo, hi),
    }


def rule_b(
    maes: Mapping[str, np.ndarray],
    threshold_s: float,
    cost: Mapping[str, float] | None = None,
) -> dict:
    """Rule B (§4.4): take the best mean; every arm within ``threshold_s``/2 of it is tied;
    among the tied, take the lowest cost (cost ties go to the lower mean, then to input order)."""
    if not maes:
        raise ValueError("rule_b needs at least one arm")
    means = {name: float(np.mean(v)) for name, v in maes.items()}
    best = min(means.values())
    tied = [n for n, m in means.items() if m <= best + threshold_s / 2 + _EPS]
    costs = {n: 0.0 for n in means} if cost is None else {n: float(cost[n]) for n in means}
    chosen = min(tied, key=lambda n: (costs[n], means[n]))
    return {"chosen": chosen, "means": means, "tied": set(tied)}
