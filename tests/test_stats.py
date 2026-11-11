"""P17 (part) Unit bootstrap, Wilcoxon, Holm and rules A and B (PROJECT_PLAN.md §10.3, §4.4)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from rulfuse import metrics as M
from rulfuse import stats as S


def test_unit_bootstrap_reproducible_by_seed():
    v = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    assert S.unit_bootstrap(v, seed=3) == S.unit_bootstrap(v, seed=3)
    assert S.unit_bootstrap(v, seed=3) != S.unit_bootstrap(v, seed=4)


def test_unit_bootstrap_matches_metrics_and_brackets_mean():
    v = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0])
    lo, hi = S.unit_bootstrap(v, seed=7)
    assert (lo, hi) == M.bootstrap_mean_ci(v, seed=7)
    assert lo < v.mean() < hi
    wide = S.unit_bootstrap(v, level=0.99, seed=7)
    assert wide[0] <= lo and wide[1] >= hi


def test_unit_bootstrap_other_statistic_and_degenerate_inputs():
    v = np.array([1.0, 2.0, 3.0, 100.0, 4.0])
    lo, hi = S.unit_bootstrap(v, statistic=np.median, seed=1)
    assert 1.0 <= lo <= hi <= 100.0
    # a statistic without an axis argument falls back to a loop
    assert S.unit_bootstrap(v, statistic=lambda s: float(s.max()), seed=1)[1] == 100.0
    assert S.unit_bootstrap(np.full(6, 2.5)) == (2.5, 2.5)
    assert S.unit_bootstrap(np.array([4.0])) == (4.0, 4.0)
    assert all(np.isnan(S.unit_bootstrap(np.array([]))))


def test_paired_bootstrap_is_bootstrap_of_differences():
    a = np.array([10.0, 12.0, 9.0, 11.0, 14.0, 8.0])
    b = a + np.array([2.0, 3.0, 1.0, 2.0, 4.0, 2.0])
    lo, hi = S.paired_bootstrap(a, b, seed=5)
    assert (lo, hi) == S.unit_bootstrap(b - a, seed=5)
    assert 1.0 <= lo <= hi <= 4.0
    assert S.paired_bootstrap(a, a + 3.0) == (3.0, 3.0)
    with pytest.raises(ValueError):
        S.paired_bootstrap(a, b[:-1])


def test_wilcoxon_matches_scipy_and_hand_p():
    a = np.zeros(10)
    b = np.arange(1.0, 11.0)
    res = S.wilcoxon_paired(a, b, "greater")
    assert res["statistic"] == 55.0
    assert res["p"] == pytest.approx(1 / 1024)  # all ten ranks positive: 1 of 2**10 sign patterns
    assert res["p"] == pytest.approx(sp_stats.wilcoxon(b - a, alternative="greater").pvalue)  # pyright: ignore[reportAttributeAccessIssue]
    assert S.wilcoxon_paired(a, b, "less")["p"] > 0.99
    assert S.wilcoxon_paired(a, b)["p"] == pytest.approx(2 / 1024)


def test_wilcoxon_mixed_signs_against_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=12)
    b = a + rng.normal(0.5, 1.0, size=12)
    for alt in ("two-sided", "greater", "less"):
        ref = sp_stats.wilcoxon(b - a, alternative=alt)
        res = S.wilcoxon_paired(a, b, alt)
        assert res["p"] == pytest.approx(ref.pvalue)  # pyright: ignore[reportAttributeAccessIssue]
        assert res["statistic"] == pytest.approx(ref.statistic)  # pyright: ignore[reportAttributeAccessIssue]


def test_wilcoxon_all_zero_differences_gives_p_one():
    a = np.array([1.0, 2.0, 3.0])
    assert S.wilcoxon_paired(a, a.copy(), "greater") == {"statistic": 0.0, "p": 1.0}


def test_holm_hand_example_and_properties():
    # sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04 -> running max
    adj = S.holm(np.array([0.01, 0.04, 0.03, 0.005]))
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(S.holm(np.array([0.5, 0.9])), [1.0, 1.0])  # capped at 1
    np.testing.assert_allclose(S.holm(np.array([0.2])), [0.2])
    assert S.holm(np.array([])).size == 0
    p = np.random.default_rng(1).uniform(size=9)
    assert np.all(S.holm(p) >= p) and np.all(S.holm(p) <= 1.0)
    # never below Bonferroni's smallest, never above Bonferroni
    assert np.all(S.holm(p) <= np.minimum(p * len(p), 1.0) + 1e-12)


def test_holm_against_loop_reference():
    p = np.array([0.001, 0.02, 0.2, 0.0004, 0.049])
    m = len(p)
    order = np.argsort(p)
    manual = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        manual[i] = run
    np.testing.assert_allclose(S.holm(p), manual)


def test_rule_a_adopts_exactly_at_threshold():
    base = np.full(10, 100.0)
    res = S.rule_a(base, base - 10.0, threshold_s=10.0)
    assert res["adopted"] and res["wins"] == 10 and res["n"] == 10
    assert res["gain_mean"] == pytest.approx(10.0)
    assert res["gain_ci"] == (pytest.approx(10.0), pytest.approx(10.0))
    assert not S.rule_a(base, base - 9.9, threshold_s=10.0)["adopted"]


def test_rule_a_needs_seven_wins():
    base = np.full(10, 100.0)
    # 7 wins of 5 s, 3 ties: mean 3.5 >= 3 -> adopted
    seven = base - np.array([5, 5, 5, 5, 5, 5, 5, 0, 0, 0], float)
    assert S.rule_a(base, seven, threshold_s=3.0)["adopted"]
    # 6 wins of 10 s, 4 ties: mean gain 6 clears 3, but only 6 wins (ties are not wins)
    six = base - np.array([10, 10, 10, 10, 10, 10, 0, 0, 0, 0], float)
    res = S.rule_a(base, six, threshold_s=3.0)
    assert res["wins"] == 6 and res["gain_mean"] == pytest.approx(6.0) and not res["adopted"]
    assert S.rule_a(base, six, threshold_s=3.0, min_wins=6)["adopted"]


def test_rule_a_rejects_large_mean_but_losses_and_shape_mismatch():
    base = np.full(10, 100.0)
    # one huge gain, nine small losses: mean gain 8 passes, wins = 1
    opt = base + 1.0
    opt[0] = 100.0 - 89.0
    res = S.rule_a(base, opt, threshold_s=3.0)
    assert res["gain_mean"] == pytest.approx(8.0) and res["wins"] == 1 and not res["adopted"]
    with pytest.raises(ValueError):
        S.rule_a(base, base[:5], threshold_s=1.0)


def test_rule_a_gain_ci_is_paired_and_seeded():
    rng = np.random.default_rng(2)
    base = rng.uniform(80, 120, 10)
    opt = base - rng.uniform(0, 20, 10)
    r1, r2 = S.rule_a(base, opt, 5.0, seed=9), S.rule_a(base, opt, 5.0, seed=9)
    assert r1["gain_ci"] == r2["gain_ci"] == S.paired_bootstrap(opt, base, seed=9)
    assert r1["gain_ci"][0] <= r1["gain_mean"] <= r1["gain_ci"][1]


def test_rule_b_ties_within_half_threshold_choose_cheaper():
    maes = {"a": np.full(4, 100.0), "b": np.full(4, 104.0), "c": np.full(4, 106.0)}
    res = S.rule_b(maes, threshold_s=10.0, cost={"a": 5.0, "b": 1.0, "c": 0.0})
    assert res["chosen"] == "b"  # c is 6 > 5 behind: not tied, however cheap
    assert res["tied"] == {"a", "b"}
    assert res["means"] == {"a": 100.0, "b": 104.0, "c": 106.0}


def test_rule_b_edges():
    maes = {"a": np.full(4, 100.0), "b": np.full(4, 105.0), "c": np.full(4, 105.5)}
    # exactly half the threshold is still tied
    res = S.rule_b(maes, threshold_s=10.0, cost={"a": 2.0, "b": 1.0, "c": 0.0})
    assert res["chosen"] == "b" and res["tied"] == {"a", "b"}
    # no cost, or equal cost: the best mean wins
    assert S.rule_b(maes, 10.0)["chosen"] == "a"
    assert S.rule_b(maes, 10.0, cost={"a": 1.0, "b": 1.0, "c": 1.0})["chosen"] == "a"
    # cost ties among the tied go to the lower mean regardless of order
    swapped = {"b": maes["b"], "a": maes["a"]}
    assert S.rule_b(swapped, 10.0, cost={"a": 3.0, "b": 3.0})["chosen"] == "a"
    # a single arm and an empty set
    assert S.rule_b({"x": np.array([1.0, 3.0])}, 10.0)["chosen"] == "x"
    with pytest.raises(ValueError):
        S.rule_b({}, 10.0)
