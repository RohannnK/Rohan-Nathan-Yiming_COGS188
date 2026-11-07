"""Per-step rows to parquet (data collection for the Week 7 monitor)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DTYPES = {
    "run_name": "string", "condition": "string", "layout": "string", "saboteur_behaviour": "string",
    "episode_seed": "int64", "t": "int32", "robot_id": "int8", "is_saboteur": "bool",
    "x": "int8", "y": "int8", "direction": "int8", "carrying_shelf_id": "Int16", "carrying_requested": "bool",
    "on_highway": "bool", "on_chokepoint": "bool", "action": "int8", "moved": "bool", "delivered": "bool",
    "picked_up": "bool", "dropped": "bool", "sabotage_active": "bool", "task_phase": "string",
    "plan_len": "float32", "plan_ms": "float32", "nodes_expanded": "float32", "replanned": "bool",
    "reward": "float32",
}


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype(DTYPES)


def write_episode(df: pd.DataFrame, data_dir: Path | str, run_name: str, seed: int) -> Path:
    """Write to <data_dir>/episodes/<run_name>/ep_<seed>.parquet."""
    path = Path(data_dir) / "episodes" / run_name / f"ep_{seed}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_run(data_dir: Path | str, run_name: str = "*") -> pd.DataFrame:
    """Concatenate every episode of one run (or all runs with the default glob) into one frame."""
    files = sorted(Path(data_dir).glob(f"episodes/{run_name}/ep_*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
