"""Training-progress figure for the DQN learner (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DQN, ASTAR, NOOP = "#2a78d6", "#eb6834", "#52514e"


def plot_dqn_progress(curves: pd.DataFrame, astar: float, noop: float, path: str | Path, window: int = 20) -> None:
    """Left: training return (rolling mean, mean +-1 std over seeds). Right: learner deliveries per evaluation
    episode against the A* robot in the same slot and a robot that never moves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=NOOP, labelsize=8)
        ax.set_xlabel("environment steps (thousands)", fontsize=8, color=NOOP)
    smooth = curves.groupby("seed")["return"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    by = smooth.groupby(curves["episode"]).agg(["mean", "std"])
    x = curves.groupby("episode")["step"].mean() / 1000
    ax1.fill_between(x, by["mean"] - by["std"], by["mean"] + by["std"], color=DQN, alpha=0.15, linewidth=0)
    ax1.plot(x, by["mean"], color=DQN, linewidth=2)
    ax1.set_title(f"DQN training return ({window}-episode mean)", fontsize=9)
    ev = curves.dropna(subset=["eval_learner_deliveries"])
    e = ev.groupby("episode").agg(step=("step", "mean"), m=("eval_learner_deliveries", "mean"),
                                  s=("eval_learner_deliveries", "std"))
    ax2.fill_between(e["step"] / 1000, e["m"] - e["s"].fillna(0), e["m"] + e["s"].fillna(0), color=DQN, alpha=0.15,
                     linewidth=0)
    ax2.plot(e["step"] / 1000, e["m"], color=DQN, linewidth=2, marker="o", markersize=3, label="DQN robot")
    ax2.axhline(astar, color=ASTAR, linewidth=2, linestyle="--", label="A* robot")
    ax2.axhline(noop, color=NOOP, linewidth=1.5, linestyle=":", label="robot that never moves")
    ax2.set_title("Deliveries by the learner's robot per evaluation episode", fontsize=9)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
