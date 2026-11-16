"""Retraining Shapley values over the sensing players (PROJECT_PLAN.md §11.1).

Players: the sensing modalities of the frozen design (here F, force signal, and T, tool photos;
D24). Usage is in every coalition; the empty coalition is the usage-only baseline B2. The value
of a coalition on one tool is ``v(S) = -window MAE (s)``, so a positive φ means the player
lowers the error.

This module is pure arithmetic over per-tool MAEs (it reads no predictions), so the same code
serves the validation tools before the freeze and the test tools after it; which one it gets
is decided by the caller (``rulfuse shapley``, ``rulfuse final``) under the test seal (D15).
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np

EMPTY = ()


def coalition_key(players) -> tuple[str, ...]:
    return tuple(sorted(players))


def shapley_values(value: dict[tuple[str, ...], float], players: list[str]) -> dict[str, float]:
    """φ_i = Σ_S |S|!(n-|S|-1)!/n! [v(S∪{i}) − v(S)] over S ⊆ N∖{i}. ``value`` maps every
    coalition (sorted tuple of players) to v; every subset must be present."""
    n = len(players)
    missing = [
        c
        for r in range(n + 1)
        for c in map(coalition_key, combinations(players, r))
        if c not in value
    ]
    if missing:
        raise ValueError(f"coalitions without a value: {missing}")
    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for r in range(len(others) + 1):
            weight = factorial(r) * factorial(n - r - 1) / factorial(n)
            for s in combinations(others, r):
                total += weight * (value[coalition_key(s + (i,))] - value[coalition_key(s)])
        phi[i] = total
    return phi


def interaction(value: dict[tuple[str, ...], float], i: str, j: str, players: list[str]) -> float:
    """Shapley interaction index of a pair (Grabisch-Roubens), averaged over the others."""
    n = len(players)
    others = [p for p in players if p not in (i, j)]
    total = 0.0
    for r in range(len(others) + 1):
        weight = factorial(r) * factorial(n - r - 2) / factorial(n - 1)
        for s in combinations(others, r):
            s = tuple(s)
            total += weight * (
                value[coalition_key(s + (i, j))]
                - value[coalition_key(s + (i,))]
                - value[coalition_key(s + (j,))]
                + value[coalition_key(s)]
            )
    return total


def per_tool_values(mae: dict[tuple[str, ...], np.ndarray], players: list[str]) -> dict:
    """Shapley values, interactions, leave-one-out and add-one values for every tool.

    ``mae[S]`` is the vector of window MAEs (one per tool, the same tool order for every
    coalition). Returns arrays per tool."""
    tools = {len(v) for v in mae.values()}
    if len(tools) != 1:
        raise ValueError("every coalition needs the same tools")
    n_tools = tools.pop()
    full = coalition_key(players)
    phi = {p: np.zeros(n_tools) for p in players}
    inter = {f"{i}x{j}": np.zeros(n_tools) for i, j in combinations(players, 2)}
    for t in range(n_tools):
        value = {c: -float(m[t]) for c, m in mae.items()}
        for p, v in shapley_values(value, players).items():
            phi[p][t] = v
        for i, j in combinations(players, 2):
            inter[f"{i}x{j}"][t] = interaction(value, i, j, players)
    loo = {
        p: np.asarray(mae[coalition_key([q for q in players if q != p])]) - np.asarray(mae[full])
        for p in players
    }
    add_one = {p: np.asarray(mae[EMPTY]) - np.asarray(mae[coalition_key([p])]) for p in players}
    return {
        "phi": phi,
        "interaction": inter,
        "leave_one_out": loo,
        "add_one": add_one,
        "gain_over_empty": np.asarray(mae[EMPTY]) - np.asarray(mae[full]),
    }


def tool_bootstrap(x: np.ndarray, n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95 % interval of the mean, resampling tools (the unit of replication)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(values: dict, tools: list[str]) -> dict:
    """Means, tool-bootstrap intervals, wins and the share of the total gain over B2."""
    gain = values["gain_over_empty"]
    out: dict = {"tools": tools, "gain_over_empty": _stat(gain), "players": {}}
    total = float(np.mean(gain))
    for p, phi in values["phi"].items():
        s = _stat(phi)
        s["share_of_gain"] = float(np.mean(phi) / total) if abs(total) > 1e-9 else float("nan")
        s["leave_one_out"] = _stat(values["leave_one_out"][p])
        s["add_one"] = _stat(values["add_one"][p])
        s["per_tool"] = {t: float(v) for t, v in zip(tools, phi, strict=True)}
        out["players"][p] = s
    out["interaction"] = {k: _stat(v) for k, v in values["interaction"].items()}
    return out


def _stat(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    lo, hi = tool_bootstrap(x)
    return {
        "mean": float(x.mean()),
        "ci95": [lo, hi],
        "positive_tools": int((x > 0).sum()),
        "tools": int(len(x)),
    }


def render(summary: dict, scope: str) -> str:
    """Markdown table; ``scope`` names the tools scored ('validation' or 'test')."""
    gain = summary["gain_over_empty"]
    lines = [
        "# Retraining Shapley values",
        "",
        f"Tools scored: **{scope}**. v(S) = −window MAE (s); usage in every coalition; empty "
        "coalition = B2. Positive = the player lowers the error. 95 % intervals resample tools.",
        "",
        f"Total gain of the full model over B2: {gain['mean']:.1f} s "
        f"[{gain['ci95'][0]:.1f}, {gain['ci95'][1]:.1f}], "
        f"better than B2 on {gain['positive_tools']}/{gain['tools']} tools.",
        "",
        "| Player | φ (s) | 95 % CI | Share of gain | Tools φ>0 | Leave-one-out (s) "
        "| Add-one (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p, s in summary["players"].items():
        lines.append(
            f"| {p} | {s['mean']:.1f} | [{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}] "
            f"| {s['share_of_gain']:.0%} | {s['positive_tools']}/{s['tools']} "
            f"| {s['leave_one_out']['mean']:.1f} | {s['add_one']['mean']:.1f} |"
        )
    lines += ["", "| Pair | Interaction (s) | 95 % CI |", "|---|---|---|"]
    for k, s in summary["interaction"].items():
        lines.append(f"| {k} | {s['mean']:.1f} | [{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}] |")
    lines += [
        "",
        "Interaction > 0: the players help more together than their separate effects add up to "
        "(complements); < 0: they overlap (substitutes).",
        "",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- study driver


def run_shapley(study, runs=None, store=None) -> dict:
    """Shapley values of a Shapley study on its VALIDATION tools (D15). The arms' players
    define the coalitions; the empty coalition is each tool's B2. Writes shapley.json and
    SHAPLEY.md in the study folder. Incomplete studies return ``{"complete": False}``."""
    import json

    from rulfuse.study import open_store, score_arm

    store = store or open_store(study.spec)
    coalitions: dict[tuple[str, ...], list[dict]] = {}
    for arm in study.arms:
        cfg = study.config(arm, runs)
        key = coalition_key(("F" if p == "signal" else p[0].upper()) for p in cfg.players)
        coalitions[key] = sorted(score_arm(study, arm, store, runs), key=lambda r: r["validation"])
    if any(not r.get("done") for rows in coalitions.values() for r in rows):
        return {"complete": False}
    full = max(coalitions, key=len)
    players = list(full)
    tools = [r["validation"] for r in coalitions[full]]
    mae = {c: np.array([r["window_mae_s"] for r in rows]) for c, rows in coalitions.items()}
    mae[EMPTY] = np.array([r["b2_window_mae_s"] for r in coalitions[full]])
    summary = summarise(per_tool_values(mae, players), tools)
    summary.update(
        {"complete": True, "scope": "validation", "study": study.name, "players_order": players}
    )
    out = study.run_dir(runs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shapley.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "SHAPLEY.md").write_text(render(summary, "validation"), encoding="utf-8")
    return summary
