"""Scores, the test seal, Shapley values and the two-modality test
(docs/OMNIFUSER_RUL_PLAN.md §7-§8; PROJECT_PLAN.md §10.3, §11, D15).

* An arm's predictions are averaged over its seeds (the ensemble), then scored per unit on
  the evaluation grid (the project's 10 s points, ``on_grid``; B6 predicts at every 1 s
  point): window MAE / RMSE / bias (RUL ≤ H, the headline) and the whole-life MAE of the
  capped target. The temporal stability (``metrics.temporal_stability``: how often and by
  how much the prediction rises, and |Δŷ + Δτ|) is measured over every prediction point.
  Summaries: mean ± sd over units, and pooled over all window points.
* Inference modes (D23, ``rulfuse.inference``): ``with_mode`` turns each fit's predictions
  of a unit, in time order, into the ``context`` mode before the seeds are averaged; the
  ``current`` mode is the predictions as written.
* Test units are sealed until the design freeze: ``role="test"`` needs ``after_freeze``
  and a date on or after ``FREEZE``.
* Shapley values over the players, with v(S) = −window MAE per test unit.
* "Are two players justified?": the best two-player arm against the best single-player arm,
  both chosen on the validation units; justified if the window MAE improves by at least
  ``MIN_GAIN_S`` and a one-sided paired Wilcoxon test over the units gives p < 0.05.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from rulfuse import metrics as M
from rulfuse.baselines import b2
from rulfuse.core.folds import leave_one_unit_out
from rulfuse.core.store import SegmentStore, grid_mask
from rulfuse.inference import CONTEXT_TIME_S, apply_mode
from rulfuse.omnifuser_rul.data import EVAL_GRID_S
from rulfuse.omnifuser_rul.train import FORCE

FREEZE = date(2026, 12, 1)
MIN_GAIN_S = 10.0  # H/150 for H = 1,500 s (PROJECT_PLAN.md §4.4)
ALPHA = 0.05


class TestSealed(RuntimeError):
    """Test-unit scores were requested before the design freeze (D15)."""


def check_seal(role: str, after_freeze: bool, today: date | None = None) -> None:
    if role != "test":
        return
    today = today or date.today()
    if not after_freeze or today < FREEZE:
        raise TestSealed(
            f"test scores are sealed until {FREEZE} (D15): pass after_freeze on or after that date"
        )


def load_predictions(folder: Path, signature: dict | None = None) -> dict[str, np.ndarray]:
    """Every fit's predictions under an arm folder; with ``signature`` (an
    ``ArmSpec.signature()``), only the fits whose ``fit.json`` records those settings, so a
    partly rerun arm never mixes fits made with other settings."""
    import json

    files = sorted(folder.glob("fold-*/seed-*/predictions.parquet"))
    if signature is not None:
        files = [
            f
            for f in files
            if (f.parent / "fit.json").is_file()
            and json.loads((f.parent / "fit.json").read_text(encoding="utf-8")).get("spec")
            == signature
        ]
    if not files:
        raise FileNotFoundError(f"no predictions under {folder}")
    tables = [pq.read_table(f) for f in files]
    columns = tables[0].column_names
    out = {
        c: np.concatenate([t.column(c).to_numpy(zero_copy_only=False) for t in tables])
        for c in columns
    }
    if "on_grid" not in out:  # predictions made at the grid's own points
        out["on_grid"] = np.ones(len(out["row"]), dtype=bool)
    return out


def with_mode(
    store: SegmentStore,
    predictions: dict[str, np.ndarray],
    mode: str,
    time_constant_s: float = CONTEXT_TIME_S,
) -> dict[str, np.ndarray]:
    """``predictions`` in inference mode ``mode`` (D23): each fit's (fold, seed) predictions
    of one unit and role, in time order, through ``rulfuse.inference.apply_mode``. Every
    other column is unchanged."""
    if mode == "current":
        return predictions
    tau = store.segments["tau_s"][predictions["row"]].astype(np.float64)
    role = np.unique(predictions["role"], return_inverse=True)[1].reshape(-1)
    unit = np.unique(predictions["unit"], return_inverse=True)[1].reshape(-1)
    fold = np.asarray(predictions["fold"], dtype=np.int64)
    seed = np.asarray(predictions["seed"], dtype=np.int64)
    order = np.lexsort((predictions["row"], tau, unit, role, seed, fold))
    key = np.stack([fold, seed, role, unit], axis=1)[order]
    starts = np.flatnonzero(np.r_[True, np.any(key[1:] != key[:-1], axis=1)])
    ends = np.r_[starts[1:], len(order)]
    values = np.asarray(predictions["prediction_s"], dtype=np.float64).copy()
    horizon = store.settings.horizon_s
    for start, end in zip(starts, ends, strict=True):
        index = order[start:end]
        values[index] = apply_mode(mode, tau[index], values[index], horizon, time_constant_s)
    return {**predictions, "prediction_s": values}


def ensemble(predictions: dict[str, np.ndarray], role: str) -> dict[str, np.ndarray]:
    """Mean over seeds for each (fold, row) of the given role."""
    keep = predictions["role"] == role
    fold, row, value = (
        predictions["fold"][keep],
        predictions["row"][keep],
        predictions["prediction_s"][keep],
    )
    unit = predictions["unit"][keep]
    on_grid = predictions["on_grid"][keep].astype(bool)
    keys, inverse, counts = np.unique(
        np.stack([fold, row], axis=1), axis=0, return_inverse=True, return_counts=True
    )
    total = np.zeros(len(keys))
    np.add.at(total, inverse.reshape(-1), value)
    first = np.zeros(len(keys), dtype=np.int64)
    first[inverse.reshape(-1)] = np.arange(len(inverse))  # any member: a row has one unit
    return {
        "fold": keys[:, 0],
        "row": keys[:, 1],
        "unit": unit[first],
        "on_grid": on_grid[first],
        "prediction_s": total / counts,
        "seeds": counts,
    }


def score_rows(store: SegmentStore, rows: np.ndarray, prediction: np.ndarray) -> dict:
    seg = store.segments
    window = seg["window"][rows].astype(bool)
    rul, y = seg["rul_s"][rows].astype(float), seg["y_s"][rows].astype(float)
    out: dict[str, float] = {"points": int(len(rows)), "window_points": int(window.sum())}
    if window.any():
        out.update(
            window_mae=M.mae(rul[window], prediction[window]),
            window_rmse=M.rmse(rul[window], prediction[window]),
            window_bias=M.bias(rul[window], prediction[window]),
        )
    out["whole_life_mae"] = M.mae(y, prediction)
    return out


def score_arm(
    store: SegmentStore,
    folder: Path,
    role: str = "validation",
    after_freeze: bool = False,
    today: date | None = None,
    signature: dict | None = None,
) -> dict:
    """Per-unit scores of the arm's seed ensemble on the evaluation grid, the temporal
    stability over every point, and the summaries (only fits with ``signature``, if given)."""
    check_seal(role, after_freeze, today)
    combined = ensemble(load_predictions(folder, signature), role)
    seg = store.segments
    per_unit = {}
    grid = combined["on_grid"].astype(bool)
    for unit in dict.fromkeys(combined["unit"].tolist()):
        mine = combined["unit"] == unit
        on = mine & grid
        scores = score_rows(store, combined["row"][on], combined["prediction_s"][on])
        rows = combined["row"][mine]
        order = np.argsort(seg["tau_s"][rows].astype(float), kind="stable")
        rows, prediction = rows[order], combined["prediction_s"][mine][order]
        uncapped = seg["y_s"][rows].astype(float) < store.settings.horizon_s - 1e-6
        stability = M.temporal_stability(prediction, seg["tau_s"][rows].astype(float), uncapped)
        per_unit[str(unit)] = {**scores, **stability, "points_all": int(len(rows))}
    pooled = score_rows(store, combined["row"][grid], combined["prediction_s"][grid])
    keys = (
        "window_mae",
        "window_rmse",
        "window_bias",
        "whole_life_mae",
        "increase_rate",
        "mean_increase_s",
        "slope_error_s",
    )
    summary = {
        k: {
            "mean": float(np.mean([s[k] for s in per_unit.values() if k in s])),
            "sd": float(np.std([s[k] for s in per_unit.values() if k in s], ddof=1))
            if len(per_unit) > 1
            else None,
            "pooled": pooled.get(k),
        }
        for k in keys
    }
    return {
        "role": role,
        "seeds": int(combined["seeds"].min()),
        "per_unit": per_unit,
        "summary": summary,
    }


def b2_by_unit(
    store: SegmentStore, role: str = "validation", grid_s: float = EVAL_GRID_S
) -> dict[str, float]:
    """B2's window MAE on each fold's validation (or test) unit, with that fold's training
    units, on the evaluation grid's points (the same points as the arms)."""
    seg = store.segments
    horizon = store.settings.horizon_s
    on_grid = grid_mask(store, grid_s)
    out = {}
    for fold in leave_one_unit_out(store.unit_names):
        unit = fold.validation if role == "validation" else fold.test
        failures = np.array(
            [store.units[u]["failure_s"] for u in fold.train if store.units[u]["failure_s"]], float
        )
        keep = (seg["unit"] == unit) & seg["window"].astype(bool) & on_grid
        tau, rul = seg["tau_s"][keep].astype(float), seg["rul_s"][keep].astype(float)
        out[unit] = M.mae(rul, b2(failures, tau, horizon))
    return out


def shapley(value: Callable[[frozenset[str]], float], players: Sequence[str]) -> dict[str, float]:
    """Exact Shapley values of a set function."""
    n = len(players)
    out = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in combinations(others, size):
                coalition = frozenset(subset)
                total += weight * (value(coalition | {player}) - value(coalition))
        out[player] = total
    return out


def paired_gain(better: Sequence[float], worse: Sequence[float]) -> dict:
    """worse − better per unit (positive: the first is better), one-sided Wilcoxon."""
    from scipy.stats import wilcoxon

    gain = np.asarray(worse, float) - np.asarray(better, float)
    p = float(wilcoxon(gain, alternative="greater").pvalue) if np.any(gain != 0) else 1.0  # pyright: ignore[reportAttributeAccessIssue]
    return {
        "mean_gain_s": float(gain.mean()),
        "units_better": int((gain > 0).sum()),
        "units_worse": int((gain < 0).sum()),
        "p_one_sided": p,
        "justified": bool(gain.mean() >= MIN_GAIN_S and p < ALPHA),
    }


def modality_report(
    window_mae: Mapping[frozenset[str], Mapping[str, float]],
    validation_mae: Mapping[frozenset[str], float],
    players: Sequence[str],
    empty_test: Mapping[str, float],
) -> dict:
    """Shapley values and the two-versus-one test.

    ``window_mae[S][unit]``: the test window MAE of the arm with players S (one per non-empty
    subset); ``validation_mae[S]``: its mean validation window MAE (chooses the compared
    arms); ``empty_test[unit]``: v(∅)'s test window MAE (B2 or the usage-only arm, whichever
    is better on validation).
    """
    units = sorted(empty_test)

    def per_unit(coalition: frozenset[str], unit: str) -> float:
        return empty_test[unit] if not coalition else window_mae[coalition][unit]

    by_unit = {u: shapley(lambda s, u=u: -per_unit(s, u), players) for u in units}
    values = {p: [by_unit[u][p] for u in units] for p in players}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(units), size=(2000, len(units)))
    phi = {
        p: {
            "mean": float(np.mean(v)),
            "ci95": [
                float(x) for x in np.quantile(np.asarray(v)[draws].mean(axis=1), [0.025, 0.975])
            ],
        }
        for p, v in values.items()
    }
    singles = [frozenset([p]) for p in players]
    pairs = [frozenset(c) for c in combinations(players, 2)]
    best_single = min(singles, key=lambda s: validation_mae[s])
    best_pair = min(pairs, key=lambda s: validation_mae[s]) if pairs else None
    report = {
        "units": units,
        "shapley": phi,
        "shapley_by_unit": by_unit,
        "best_single": sorted(best_single),
    }
    if best_pair is not None:
        report["best_pair"] = sorted(best_pair)
        report["pair_vs_single"] = paired_gain(
            [window_mae[best_pair][u] for u in units], [window_mae[best_single][u] for u in units]
        )
        photo = [p for p in players if p != FORCE]
        if FORCE in players and len(photo) >= 2:
            photo_pairs = [frozenset(c) for c in combinations(photo, 2)]
            force_pairs = [frozenset([FORCE, p]) for p in photo]
            best_photos = min(photo_pairs, key=lambda s: validation_mae[s])
            best_force_photo = min(force_pairs, key=lambda s: validation_mae[s])
            report["force_photo_vs_photos"] = {
                "force_photo": sorted(best_force_photo),
                "photos": sorted(best_photos),
                **paired_gain(
                    [window_mae[best_force_photo][u] for u in units],
                    [window_mae[best_photos][u] for u in units],
                ),
            }
        full = frozenset(players)
        if len(players) > 2:
            report["all_vs_pair"] = paired_gain(
                [window_mae[full][u] for u in units], [window_mae[best_pair][u] for u in units]
            )
    return report
