"""P2 Causal trim (PROJECT_PLAN.md §5.2, §15)."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from rulfuse import paths
from rulfuse.adapters.nonastreda.dataset import trimmed_bounds
from rulfuse.adapters.nonastreda.force_store import ForceStore
from rulfuse.adapters.nonastreda.trim import span_bounds_causal, span_bounds_omnifuser, trim_bounds

OMNIFUSER_RECORDS = (
    paths.REPO_ROOT.parent / "OmniFuser" / "experiment_results" / "eda" / "records.csv"
)


def synthetic_pass(lead_s: float, cut_s: float, out_s: float, seed: int = 0) -> np.ndarray:
    """Idle at |F| ~ 5 N, cutting at |F| ~ 120 N with noise, idle again."""
    rng = np.random.default_rng(seed)
    parts = []
    for seconds, level in ((lead_s, 5.0), (cut_s, 120.0), (out_s, 5.0)):
        n = int(seconds * 1000)
        block = np.zeros((n, 3), np.float32)
        block[:, 0] = level + rng.normal(0, 0.05 * level, n)
        parts.append(block)
    return np.concatenate(parts)


def test_short_lead_in_and_run_out_are_removed():
    force = synthetic_pass(1.0, 60.0, 1.0)
    start, end = span_bounds_causal(force)
    assert abs(start - 1000) <= 20 and abs(end - 61000) <= 20


def test_long_lead_in_is_the_documented_limit():
    # More than half of the 5 s reference idle: the reference sits at the idle level and
    # nothing is trimmed at the start (EDA.md §5 documents the same limit for OmniFuser).
    force = synthetic_pass(3.0, 60.0, 0.0)
    assert span_bounds_causal(force)[0] == 0


def test_causal_start_does_not_depend_on_the_rest_of_the_pass():
    force = synthetic_pass(1.0, 60.0, 1.0)
    start_full = span_bounds_causal(force)[0]
    for keep_s in (6, 10, 30):
        assert span_bounds_causal(force[: keep_s * 1000])[0] == start_full


def test_omnifuser_reference_depends_on_the_pass_length():
    # With a 2 s lead-in: 5 % of a 200 s pass is 10 s (mostly cutting, trimmed), 5 % of a
    # 30 s pass is 1.5 s (all idle, not trimmed): the start depends on the future length.
    long_pass = synthetic_pass(2.0, 198.0, 0.0)
    assert span_bounds_omnifuser(long_pass)[0] > 1900
    assert span_bounds_omnifuser(long_pass[:30000])[0] == 0
    assert span_bounds_causal(long_pass)[0] == span_bounds_causal(long_pass[:30000])[0]


def test_degenerate_inputs():
    assert span_bounds_causal(np.zeros((10, 3), np.float32)) == (0, 10)
    assert span_bounds_causal(np.zeros((5000, 3), np.float32)) == (0, 5000)
    with pytest.raises(ValueError):
        trim_bounds(np.zeros((100, 3)), "nope")


@pytest.mark.data
def test_vendored_omnifuser_rule_reproduces_omnifuser():
    if not OMNIFUSER_RECORDS.is_file():
        pytest.skip("OmniFuser's records.csv is not available")
    theirs = {
        r["record_id"]: (int(r["trim_start"]), int(r["trim_end"]))
        for r in csv.DictReader(OMNIFUSER_RECORDS.open())
    }
    ours = trimmed_bounds(ForceStore.open(paths.force_store_dir()), "omnifuser")
    assert ours == theirs


@pytest.mark.data
def test_causal_trim_agrees_with_omnifuser(data):
    """Acceptance P2: start within 200 samples of OmniFuser's on >= 95 % of passes."""
    omni = trimmed_bounds(data.store, "omnifuser")
    diffs = np.array([data.bounds[k][0] - omni[k][0] for k in omni])
    assert np.mean(np.abs(diffs) <= 200) >= 0.95
