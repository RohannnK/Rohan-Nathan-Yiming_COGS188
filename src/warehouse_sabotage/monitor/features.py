"""Sliding-window behavioural features (partial evidence of hidden intent): the 12 proposal features.

Computed from per-robot step rows in the project.md section 7 log schema. The same functions score logged episodes
(monitor/dataset.py) and live episodes (monitor/quarantine.py), so training and deployment cannot drift apart.
Only observable columns are read (positions, loads, actions, moves and honest robots' planned tiles); never
is_saboteur, sabotage_active, task_phase or the planner diagnostics, which would leak the label.

Definitions (window of W steps, one value per robot):
  corridor_frac          fraction of steps on a corridor tile (1-wide chokepoint that is not a goal or goal access tile)
  choke_steps            steps on any chokepoint tile (corridor, goal or goal access)
  idle_ticks             steps in which the position did not change
  unrequested_carried    steps spent carrying a shelf that is not requested
  requested_undelivered  requested shelves picked up in the window minus deliveries in the window (at least 0)
  deliveries             deliveries completed in the window
  next_to_stalled        steps next to another robot that has not moved for STALL_STEPS steps on a highway tile
  aisle_blocking         steps on a tile in another robot's planned path (planned_cells, next 8 tiles)
  path_efficiency        progress towards the current target per step (shortest-path distance covered / steps)
  goal_distance          mean Manhattan distance to the nearest goal
  drop_displacement      mean Manhattan distance between pick-up and drop tile of the shelves dropped in the window
  action_entropy         Shannon entropy (bits) of the action distribution
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..env.layouts import chokepoints
from ..planning.grid_graph import UNREACHABLE, bfs_distances
from ..planning.heuristics import manhattan

FEATURES = ("corridor_frac", "choke_steps", "idle_ticks", "unrequested_carried", "requested_undelivered",
            "deliveries", "next_to_stalled", "aisle_blocking", "path_efficiency", "goal_distance",
            "drop_displacement", "action_entropy")
STALL_STEPS = 3  # a robot that has not moved for this many steps (including the current one) is stalled


def corridor_tiles(highways: np.ndarray, goals) -> frozenset[tuple[int, int]]:
    """Chokepoints that are not goals or goal access tiles (the 1-wide corridors)."""
    return frozenset(c for c in chokepoints(highways, goals) if min(manhattan(c, g) for g in goals) > 1)


def _cells(planned) -> set[tuple[int, int]]:
    return {(int(c[0]), int(c[1])) for c in planned} if planned is not None else set()


def step_table(ep: pd.DataFrame, highways: np.ndarray, goals) -> pd.DataFrame:
    """Per-step quantities behind the features, one row per (robot, t), sorted by robot_id then t.

    ``ep`` is one episode in the log schema (planned_cells, target_x and target_y are optional).
    """
    ep = ep.sort_values(["robot_id", "t"]).reset_index(drop=True)
    corridor = corridor_tiles(highways, goals)
    goals = list(goals)
    xy = list(zip(ep["x"], ep["y"]))
    carrying = ep["carrying_shelf_id"].notna().to_numpy()
    requested = ep["carrying_requested"].to_numpy(bool)
    g = ep.groupby("robot_id", sort=False)

    out = pd.DataFrame({"robot_id": ep["robot_id"], "t": ep["t"], "action": ep["action"]})
    out["corridor"] = [c in corridor for c in xy]
    out["choke"] = ep["on_chokepoint"].to_numpy(bool)
    out["idle"] = ~ep["moved"].to_numpy(bool)
    out["unrequested"] = carrying & ~requested
    out["delivered"] = ep["delivered"].to_numpy(bool)
    nxt_requested = g["carrying_requested"].shift(-1, fill_value=False).to_numpy(bool)
    out["picked_requested"] = ep["picked_up"].to_numpy(bool) & nxt_requested
    out["goal_distance"] = [min(manhattan(c, gl) for gl in goals) for c in xy]

    # stalled robots, then who stands next to one
    still = pd.Series(~ep["moved"].to_numpy(bool)).groupby(ep["robot_id"]).rolling(STALL_STEPS).min()
    stalled = still.fillna(0).to_numpy(bool) & ep["on_highway"].to_numpy(bool)  # sorted like ep: same grouping order
    stalled_at = {(t, r): s for t, r, s in zip(ep["t"], ep["robot_id"], stalled)}
    pos = {(t, r): c for t, r, c in zip(ep["t"], ep["robot_id"], xy)}
    robots = ep["robot_id"].unique()
    planned = ep["planned_cells"] if "planned_cells" in ep else [None] * len(ep)
    path_cells = {(t, r): _cells(p) for t, r, p in zip(ep["t"], ep["robot_id"], planned)}
    near, block = [], []
    for t, r, c in zip(ep["t"], ep["robot_id"], xy):
        others = [o for o in robots if o != r]
        near.append(any(stalled_at.get((t, o), False) and manhattan(pos[(t, o)], c) == 1 for o in others))
        block.append(any(c in path_cells.get((t, o), ()) for o in others))
    out["next_to_stalled"], out["aisle_blocking"] = near, block

    # shelf drop displacement: distance from the pick-up tile (carried forward within a robot) to the drop tile
    pick = pd.DataFrame({"x": ep["x"].where(ep["picked_up"]), "y": ep["y"].where(ep["picked_up"])})
    pick = pick.groupby(ep["robot_id"]).ffill()
    disp = (ep["x"] - pick["x"]).abs() + (ep["y"] - pick["y"]).abs()
    out["drop_displacement"] = disp.where(ep["dropped"])

    # distance to the current target (BFS ignoring robots; loaded robots use highways only)
    dist = np.full(len(ep), np.nan)
    if "target_x" in ep:
        cache = {}
        for i, (c, loaded, tx, ty) in enumerate(zip(xy, carrying, ep["target_x"], ep["target_y"])):
            if pd.isna(tx):
                continue
            key = (int(tx), int(ty), bool(loaded))
            if key not in cache:
                mask = highways.astype(bool) if loaded else np.ones(highways.shape, dtype=bool)
                mask[key[1], key[0]] = True
                cache[key] = bfs_distances(mask, [key[:2]])
            d = cache[key][c[1], c[0]]
            dist[i] = d if d < UNREACHABLE else np.nan
        out["target"] = [None if pd.isna(a) else (int(a), int(b)) for a, b in zip(ep["target_x"], ep["target_y"])]
    else:
        out["target"] = None
    out["target_distance"] = dist
    return out


def _efficiency(target, dist) -> float:
    """Progress towards the target held at the window start, per step, over the steps that kept that target."""
    if target[0] is None:
        return 0.0
    n = 1
    while n < len(target) and target[n] == target[0]:
        n += 1
    if n < 2 or np.isnan(dist[0]) or np.isnan(dist[n - 1]):
        return 0.0
    return float(np.clip((dist[0] - dist[n - 1]) / (n - 1), -1.0, 1.0))


def _entropy(actions) -> float:
    p = np.bincount(actions, minlength=5) / len(actions)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def window_features(w: pd.DataFrame) -> dict:
    """The 12 features for one robot's window of consecutive step rows (a slice of ``step_table``)."""
    n = len(w)
    drops = w["drop_displacement"].dropna()
    return {
        "corridor_frac": w["corridor"].sum() / n,
        "choke_steps": int(w["choke"].sum()),
        "idle_ticks": int(w["idle"].sum()),
        "unrequested_carried": int(w["unrequested"].sum()),
        "requested_undelivered": max(int(w["picked_requested"].sum() - w["delivered"].sum()), 0),
        "deliveries": int(w["delivered"].sum()),
        "next_to_stalled": int(w["next_to_stalled"].sum()),
        "aisle_blocking": int(w["aisle_blocking"].sum()),
        "path_efficiency": _efficiency(list(w["target"]), w["target_distance"].to_numpy()),
        "goal_distance": float(w["goal_distance"].mean()),
        "drop_displacement": float(drops.mean()) if len(drops) else 0.0,
        "action_entropy": _entropy(w["action"].to_numpy(int)),
    }


def window_table(ep: pd.DataFrame, highways: np.ndarray, goals, window: int = 20, stride: int = 10,
                 ends=None) -> pd.DataFrame:
    """One row per robot and window: robot_id, t_end (last step of the window) and the 12 features.

    Windows are full-length and end at t = window - 1, window - 1 + stride, ... but never on the last step of the
    episode, because a live monitor only completes step t when it sees the state at t + 1; ``ends`` overrides this
    (the live monitor asks for the latest one). Label columns (is_saboteur, sabotage_window) are copied from the
    log when it has them, for training and evaluation only.
    """
    steps = step_table(ep, highways, goals)
    ep = ep.sort_values(["robot_id", "t"]).reset_index(drop=True)
    horizon = int(ep["t"].max()) + 1
    ends = range(window - 1, horizon - 1, stride) if ends is None else ends
    rows = []
    for rid, s in steps.groupby("robot_id", sort=False):
        s = s.reset_index(drop=True)
        labels = ep[ep["robot_id"] == rid].reset_index(drop=True)
        for end in ends:
            lo = end - window + 1
            row = {"robot_id": int(rid), "t_end": int(end), **window_features(s.iloc[lo:end + 1])}
            if "is_saboteur" in labels:
                row["is_saboteur"] = bool(labels["is_saboteur"].iat[0])
                row["sabotage_window"] = bool(labels["sabotage_active"].iloc[lo:end + 1].any())
            rows.append(row)
    return pd.DataFrame(rows)
