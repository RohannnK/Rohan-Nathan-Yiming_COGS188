"""Episode and condition metrics: throughput, deadlock rate, planning cost (performance measures, section 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def longest_gap(delivery_steps, horizon: int) -> int:
    """Longest run of consecutive steps with no delivery (requests are always outstanding in rware)."""
    edges = np.concatenate([[-1], np.sort(delivery_steps), [horizon]])
    return int(np.diff(edges).max() - 1)


def episode_metrics(df: pd.DataFrame, deadlock_L: int = 100) -> dict:
    """Metrics for one episode's rows."""
    horizon = int(df["t"].max()) + 1
    honest = df[~df["is_saboteur"]]
    delivery_steps = df.loc[df["delivered"], "t"].unique()
    planned = df[df["plan_ms"].notna()]
    gap = longest_gap(delivery_steps, horizon)
    return {
        "run_name": df["run_name"].iat[0], "condition": df["condition"].iat[0], "layout": df["layout"].iat[0],
        "saboteur_behaviour": df["saboteur_behaviour"].iat[0], "episode_seed": int(df["episode_seed"].iat[0]),
        "steps": horizon,
        "deliveries": int(honest["delivered"].sum()),  # throughput D: honest deliveries only
        "saboteur_deliveries": int(df.loc[df["is_saboteur"], "delivered"].sum()),
        "longest_no_delivery": gap,
        "deadlock": gap >= deadlock_L,
        "plan_calls": len(planned),
        "plan_ms_per_call": float(planned["plan_ms"].mean()) if len(planned) else np.nan,
        "plan_ms_per_step": float(df.groupby("t")["plan_ms"].sum().mean()),
        "nodes_per_call": float(planned["nodes_expanded"].mean()) if len(planned) else np.nan,
        "replans": int(honest["replanned"].sum()),
        "sabotage_steps": int(df["sabotage_active"].sum()),
    }


SUMMARY_COLS = ["deliveries", "deadlock", "longest_no_delivery", "plan_ms_per_call", "plan_ms_per_step",
                "nodes_per_call", "replans", "sabotage_steps"]


def condition_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, layout): mean and std of episode metrics, plus D / D(no_saboteur) per layout."""
    g = episodes.groupby(["condition", "layout"], sort=False)
    out = g[SUMMARY_COLS].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out = out.rename(columns={"deadlock_mean": "deadlock_rate", "deadlock_std": "deadlock_rate_std"})
    out.insert(0, "episodes", g.size())
    out = out.reset_index()
    base = out[out["condition"] == "no_saboteur"].set_index("layout")["deliveries_mean"]
    out["ratio_to_no_saboteur"] = out["deliveries_mean"] / out["layout"].map(base)
    return out
