"""Hand-made episode logs for monitor tests (log schema of project.md section 7, no simulator needed)."""
import numpy as np
import pandas as pd

from warehouse_sabotage.env.layouts import LAYOUTS, chokepoints, parse_layout

HIGHWAYS, GOALS = parse_layout(LAYOUTS["tiny_choke"])
CHOKE = chokepoints(HIGHWAYS, GOALS)


def episode(paths, actions=None, seed=1, saboteur=None, planned=None, **cols) -> pd.DataFrame:
    """paths: one list of (x, y) per robot (position at each t). Everything else defaults to 'nothing happens'.

    ``moved`` is derived from the positions; ``actions``/``planned`` map robot -> list per step; ``cols`` override
    columns for all rows (for example carrying_requested).
    """
    rows = []
    for rid, path in enumerate(paths):
        for t, (x, y) in enumerate(path):
            nxt = path[t + 1] if t + 1 < len(path) else (x, y)
            rows.append({
                "run_name": "test", "condition": "test", "layout": "tiny_choke", "saboteur_behaviour": "none",
                "episode_seed": seed, "t": t, "robot_id": rid, "is_saboteur": rid == saboteur,
                "x": x, "y": y, "direction": 1, "carrying_shelf_id": None, "carrying_requested": False,
                "on_highway": bool(HIGHWAYS[y, x]), "on_chokepoint": (x, y) in CHOKE,
                "action": (actions or {}).get(rid, [0] * len(path))[t], "moved": (x, y) != nxt,
                "delivered": False, "picked_up": False, "dropped": False, "sabotage_active": False,
                "planned_cells": (planned or {}).get(rid, [[]] * len(path))[t],
            })
    df = pd.DataFrame(rows)
    for k, v in cols.items():
        df[k] = v
    df["carrying_shelf_id"] = df["carrying_shelf_id"].astype("Int16")
    return df


def synthetic_runs(n_seeds=24, steps=120, seed=0) -> pd.DataFrame:
    """Saboteur-vs-honest toy logs: honest robots shuttle along row 0 and row 9; the saboteur parks in a corridor
    from t = 60 (sabotage_active) in the 'sabotage' run. The 'clean' run has the same seeds and four honest robots."""
    rng = np.random.default_rng(seed)
    eps = []
    for s in range(n_seeds):
        k = int(rng.integers(4))
        lanes = [[(x, 0) for x in range(0, 10)] + [(x, 0) for x in range(9, -1, -1)] for _ in range(4)]
        paths = [(lane * 10)[int(rng.integers(5)):][:steps] for lane in lanes]
        for run, sab in (("sabotage", k), ("clean", None)):
            p = [list(q) for q in paths]
            if sab is not None:
                p[sab] = p[sab][:60] + [(3, 5)] * (steps - 60)
            df = episode(p, seed=1000 + s, saboteur=sab)
            df["run_name"], df["condition"] = run, run
            if sab is not None:
                df.loc[(df["robot_id"] == sab) & (df["t"] >= 60), "sabotage_active"] = True
            eps.append(df)
    return pd.concat(eps, ignore_index=True)
