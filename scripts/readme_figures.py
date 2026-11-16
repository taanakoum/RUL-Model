"""The README's figures, in a light and a dark version (validation tools only, D15).

Numbers come from the study reports (runs/gate2a-fix, gate2b-fix, gate3 REPORT.md). Usage:

    python scripts/readme_figures.py          # writes docs/readme/*-{light,dark}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/readme")
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": {"text": "#14213D", "muted": "#5b6275", "grid": "#e3e2dc", "a": "#17695F",
              "b": "#E9A03B", "c": "#9aa3b5", "bad": "#A8372A"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3a37", "a": "#2fa595",
             "b": "#f0b45a", "c": "#7d869a", "bad": "#d9695a"},
}

TOOLS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
MODEL = [417, 387, 275, 689, 397, 272, 298, 223, 320, 402]
CONSTANT = [378, 447, 377, 659, 377, 376, 378, 376, 377, 375]
USAGE = [621, 228, 464, 775, 748, 240, 328, 304, 309, 631]


def style(ax, t):
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=10)
    ax.xaxis.label.set_color(t["muted"])
    ax.yaxis.label.set_color(t["muted"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=170, transparent=True, bbox_inches="tight")
    plt.close(fig)


for mode, t in THEMES.items():
    # 1. per validation tool
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    x = np.arange(len(TOOLS))
    w = 0.27
    ax.bar(x - w, MODEL, w, color=t["a"], label="RUL-Fuse (F0)")
    ax.bar(x, CONSTANT, w, color=t["c"], label="Constant guess (B0)")
    ax.bar(x + w, USAGE, w, color=t["b"], label="Usage only (B2)")
    ax.set_xticks(x, TOOLS)
    ax.set_ylabel("Window MAE (s), lower is better")
    style(ax, t)
    leg = ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    save(fig, "per_tool", mode)

    # 2. design decisions
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [2, 3]})
    ax = axes[0]
    labels = ["Per-tool\nmean", "Pooled\npoints", "Without\nT4"]
    f1, f0 = [389.1, 325.5, 310.8], [368.0, 335.8, 332.3]
    xx = np.arange(3)
    ax.bar(xx - 0.19, f1, 0.38, color=t["c"], label="F1 transformer")
    ax.bar(xx + 0.19, f0, 0.38, color=t["a"], label="F0 late fusion")
    ax.set_xticks(xx, labels)
    ax.set_ylim(250, 420)
    ax.set_title("Fusion family", color=t["text"], fontsize=11, loc="left")
    ax.set_ylabel("Window MAE (s)")
    style(ax, t)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    ax = axes[1]
    arms = ["default", "no trajectory\naug", "no phase A", "tokens\nrelative", "tokens\nabsolute"]
    vals = [368.0, 358.5, 402.1, 367.8, 367.3]
    colors = [t["a"], t["a"], t["bad"], t["a"], t["a"]]
    ax.bar(np.arange(5), vals, 0.6, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, f"{v:.0f}", ha="center", color=t["text"], fontsize=9)
    ax.set_xticks(np.arange(5), arms, fontsize=8)
    ax.set_ylim(250, 430)
    ax.set_title("Design options (F0)", color=t["text"], fontsize=11, loc="left")
    style(ax, t)
    fig.tight_layout()
    save(fig, "decisions", mode)
