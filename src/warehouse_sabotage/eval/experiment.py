"""Run a config across seeds and conditions (experimental design), plus the planner comparison.

Every condition runs on the same explicit seeds, so conditions are paired: seed s gives the same spawns and
initial requests in every condition.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..env.layouts import LAYOUTS, parse_layout
from ..planning.search import search
from .logging import rows_to_frame, write_episode
from .metrics import episode_metrics
from .rollout import run_episode

log = logging.getLogger(__name__)


def episode_specs(cfg: dict, smoke: bool = False):
    """Yield (spec, seeds) per condition x layout. Smoke mode: first `smoke.episodes` seeds, `smoke.max_steps`."""
    seeds = list(cfg["seeds"])
    env = dict(cfg.get("env", {}))
    name = cfg["name"]
    if smoke:
        seeds = seeds[: cfg["smoke"]["episodes"]]
        env["max_steps"] = cfg["smoke"]["max_steps"]
        name += "_smoke"
    for condition, kind in cfg["conditions"].items():
        for layout in cfg["layouts"]:
            yield {
                "run_name": f"{name}__{condition}__{layout}", "condition": condition, "layout": layout,
                "kind": kind, "env": env, "planner": cfg.get("planner", {}), "saboteur": cfg.get("saboteur", {}),
            }, seeds


def run_experiment(cfg: dict, smoke: bool = False, data_dir="data", results_dir="results") -> pd.DataFrame:
    """Run every episode, write parquet logs and per-run episodes.csv; return the per-episode metrics frame."""
    episodes = []
    for spec, seeds in episode_specs(cfg, smoke):
        t0 = time.perf_counter()
        L = cfg.get("deadlock_L", 100)
        L = L[spec["layout"]] if isinstance(L, dict) else L
        run_eps = []
        for seed in seeds:
            df = rows_to_frame(run_episode(spec, seed))
            write_episode(df, data_dir, spec["run_name"], seed)
            run_eps.append(episode_metrics(df, L))
        run_df = pd.DataFrame(run_eps)
        out = Path(results_dir) / spec["run_name"]
        out.mkdir(parents=True, exist_ok=True)
        run_df.to_csv(out / "episodes.csv", index=False)
        episodes.append(run_df)
        log.info("%s: %d episodes in %.1fs, D=%.1f", spec["run_name"], len(seeds),
                 time.perf_counter() - t0, run_df["deliveries"].mean())
    return pd.concat(episodes, ignore_index=True)


def compare_planners(n_pairs: int = 100, seed: int = 0, layouts=("tiny", "tiny_choke"),
                     algorithms=("bfs", "ucs", "greedy", "astar")) -> pd.DataFrame:
    """BFS vs UCS vs greedy vs A* on random start/goal pairs on the reset-time maps.

    Loaded robots may only use highways (every slot holds a shelf at reset); unloaded robots may use any tile.
    Returns one row per (layout, load, algorithm, pair).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for layout in layouts:
        highways, _ = parse_layout(LAYOUTS[layout])
        for load in ("unloaded", "loaded"):
            mask = highways.astype(bool) if load == "loaded" else np.ones_like(highways, dtype=bool)
            cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(mask))]
            for pair in range(n_pairs):
                i, j = rng.choice(len(cells), 2, replace=False)
                start, goal = (*cells[i], int(rng.integers(4))), cells[j]
                for alg in algorithms:
                    r = search(start, goal, mask, alg)
                    rows.append({"layout": layout, "load": load, "algorithm": alg, "pair": pair,
                                 "found": r.found, "cost": r.cost, "nodes_expanded": r.nodes_expanded, "ms": r.ms})
    df = pd.DataFrame(rows)
    optimal = df[df["algorithm"] == "ucs"].set_index(["layout", "load", "pair"])["cost"]
    df["optimal"] = df["cost"].to_numpy() == optimal.loc[list(zip(df["layout"], df["load"], df["pair"]))].to_numpy()
    return df


def summarise_planners(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["layout", "load", "algorithm"], sort=False).agg(
        pairs=("pair", "size"), found_rate=("found", "mean"), optimal_rate=("optimal", "mean"),
        cost_mean=("cost", "mean"), cost_std=("cost", "std"),
        nodes_mean=("nodes_expanded", "mean"), nodes_std=("nodes_expanded", "std"),
        ms_mean=("ms", "mean"), ms_std=("ms", "std"))
    return out.reset_index()
