"""Static report figures (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = {"sarsa": "#2a78d6", "q_learning": "#eb6834"}  # categorical slots 1 and 2
LABELS = {"sarsa": "SARSA", "q_learning": "Q-learning"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_learning_curves(curves: pd.DataFrame, metric: str, ylabel: str, path: str | Path, gamma: float,
                         window: int = 20, ylim: dict[str, tuple[float, float]] | None = None) -> None:
    """Small multiples (rows: bump_prob, columns: epsilon schedule) of a per-episode metric, rolling mean over
    `window` episodes, mean line and +-1 std band across seeds, one line per method. ``ylim`` maps a schedule to
    its y range; each column has its own y scale."""
    d = curves[curves["gamma"] == gamma]
    bumps, scheds = sorted(d["bump_prob"].unique()), list(dict.fromkeys(d["schedule"]))
    fig, axes = plt.subplots(len(bumps), len(scheds), figsize=(4.2 * len(scheds), 2.8 * len(bumps)),
                             sharex=True, sharey="col", squeeze=False)
    for i, bump in enumerate(bumps):
        for j, sched in enumerate(scheds):
            ax = axes[i, j]
            _style(ax)
            for method, colour in SERIES.items():
                c = d[(d["bump_prob"] == bump) & (d["schedule"] == sched) & (d["method"] == method)]
                smooth = c.groupby("seed")[metric].transform(lambda s: s.rolling(window, min_periods=1).mean())
                stats = smooth.groupby(c["episode"]).agg(["mean", "std"])
                ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                                color=colour, alpha=0.15, linewidth=0)
                ax.plot(stats.index, stats["mean"], color=colour, linewidth=2, label=LABELS[method])
            ax.set_title(f"blocker bump p = {bump}, {sched} epsilon", fontsize=9, color=INK)
            if ylim and sched in ylim:
                ax.set_ylim(*ylim[sched])
            if i == len(bumps) - 1:
                ax.set_xlabel("training episode", fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_paths(layout: tuple[str, ...], paths: dict[str, list[tuple[int, int]]], path: str | Path,
               title: str) -> None:
    """Draw the gridworld (shelves dark, blocker lane hatched, aisle tinted) with one greedy path per method."""
    h, w = len(layout), len(layout[0])
    fig, ax = plt.subplots(figsize=(6, 2.9))
    fills = {"x": "#52514e", "b": "#f3d3c6", "a": "#f5f4f0", "S": "#ffffff", "G": "#ffffff", ".": "#ffffff"}
    for y, row in enumerate(layout):
        for x, c in enumerate(row):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=fills[c], edgecolor=GRID, linewidth=1,
                                       hatch="///" if c == "b" else None))
            if c in "SG":
                ax.text(x + 0.5, y + 0.5, c, ha="center", va="center", fontsize=10, color=INK, weight="bold")
    offsets = {"sarsa": -0.12, "q_learning": 0.12}
    for method, tiles in paths.items():
        xs = [x + 0.5 + offsets[method] for x, _ in tiles]
        ys = [y + 0.5 + offsets[method] for _, y in tiles]
        ax.plot(xs, ys, color=SERIES[method], linewidth=2, marker="o", markersize=3,
                label=f"{LABELS[method]} ({len(tiles) - 1} steps)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
