"""Window-level monitor dataset from episode logs, split by episode (supervised learning without leakage).

Every row is one robot over one window. ``is_saboteur`` is the robot-level label and ``sabotage_window`` marks the
windows in which sabotage was actually happening. The other columns are metadata and the 12 features.

Seeds are shared by every condition and layout (paired design), so episodes with the same seed have the same spawns,
requests and saboteur id. Splitting therefore groups by ``episode_seed``, not by run, to keep related episodes together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..env.layouts import LAYOUTS, parse_layout
from .features import window_table

META = ["run_name", "condition", "layout", "saboteur_behaviour", "episode_seed"]


def build_windows(episodes: pd.DataFrame, window: int = 20, stride: int = 10) -> pd.DataFrame:
    """Features and labels for every episode in a log frame (one or more runs, see eval.logging.load_run)."""
    parts = []
    for _, ep in episodes.groupby(["run_name", "episode_seed"], sort=False):
        highways, goals = parse_layout(LAYOUTS[ep["layout"].iat[0]])
        w = window_table(ep, highways, goals, window, stride)
        for c in META:
            w.insert(META.index(c), c, ep[c].iat[0])
        active = ep.loc[ep["sabotage_active"], "t"]
        w["t_first_sabotage"] = active.min() if len(active) else np.nan  # for time-to-detection, never a feature
        w["horizon"] = int(ep["t"].max()) + 1
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def split_by_seed(windows: pd.DataFrame, fractions=(0.6, 0.2, 0.2), seed: int = 0) -> pd.Series:
    """'train' / 'val' / 'test' label per row; all windows of one episode_seed land in the same split."""
    seeds = np.sort(windows["episode_seed"].unique())
    seeds = np.random.default_rng(seed).permutation(seeds)
    cut = np.cumsum([round(f * len(seeds)) for f in fractions[:-1]])
    names = np.empty(len(seeds), dtype=object)
    for name, part in zip(("train", "val", "test"), np.split(np.arange(len(seeds)), cut)):
        names[part] = name
    return windows["episode_seed"].map(dict(zip(seeds, names)))
