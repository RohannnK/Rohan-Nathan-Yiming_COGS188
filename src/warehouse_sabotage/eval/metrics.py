"""Episode and condition metrics: throughput, deadlock rate, planning cost (performance measures, section 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def longest_gap(delivery_steps, horizon: int) -> int:
    """Longest run of consecutive steps with no delivery (requests are always outstanding in rware)."""
    edges = np.concatenate([[-1], np.sort(delivery_steps), [horizon]])
    return int(np.diff(edges).max() - 1)


def flag_metrics(df: pd.DataFrame) -> dict:
    """Closed-loop monitor outcome of one episode: first flag of the saboteur, honest robots flagged, honest robots."""
    sab = df[df["is_saboteur"]]
    active = sab.loc[sab["sabotage_active"], "t"]
    flagged = df[df["flagged"]] if "flagged" in df else df.iloc[:0]  # logs written before Week 7 have no column
    return {
        "t_first_sabotage": float(active.min()) if len(active) else np.nan,
        "t_flag_saboteur": float(flagged.loc[flagged["is_saboteur"], "t"].min()) if flagged["is_saboteur"].any() else np.nan,
        "false_flags": int(flagged.loc[~flagged["is_saboteur"], "robot_id"].nunique()),
        "honest_robots": int(df.loc[~df["is_saboteur"], "robot_id"].nunique()),
    }


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
        **flag_metrics(df),
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


def window_metrics(y_true, score, flagged) -> dict:
    """Window-level precision, recall, F1 and ROC-AUC (monitor performance, section 8)."""
    y_true = np.asarray(y_true, bool)
    kw = {"zero_division": 0}
    return {"precision": precision_score(y_true, flagged, **kw), "recall": recall_score(y_true, flagged, **kw),
            "f1": f1_score(y_true, flagged, **kw),
            "roc_auc": roc_auc_score(y_true, score) if 0 < y_true.sum() < len(y_true) else np.nan}


def first_flag(windows: pd.DataFrame, flag: str = "flag", persist: int = 1) -> pd.DataFrame:
    """Per robot-episode: first window end t_flag with ``persist`` consecutive flagged windows (NaN if never)."""
    rows = []
    for key, w in windows.sort_values("t_end").groupby(["run_name", "episode_seed", "robot_id"], sort=False):
        run = w[flag].astype(int).rolling(persist).sum().eq(persist)
        rows.append({"run_name": key[0], "episode_seed": key[1], "robot_id": key[2],
                     "is_saboteur": bool(w["is_saboteur"].iat[0]), "t_first_sabotage": w["t_first_sabotage"].iat[0],
                     "horizon": w["horizon"].iat[0], "t_flag": w["t_end"][run.to_numpy()].min() if run.any() else np.nan})
    return pd.DataFrame(rows)


def detection_metrics(flags: pd.DataFrame) -> dict:
    """Time-to-detection, detection rate and robot-episode false-positive rate from ``first_flag`` output.

    TTD = t_flag - t_first_sabotage; a saboteur that is never flagged counts as horizon - t_first_sabotage. A saboteur
    flagged before its first sabotage step is an early detection: TTD is floored at 0 and ``early_flag_rate`` reports them.
    """
    sab = flags[flags["is_saboteur"] & flags["t_first_sabotage"].notna()]
    ttd = (sab["t_flag"].fillna(sab["horizon"]) - sab["t_first_sabotage"]).clip(lower=0)
    honest = flags[~flags["is_saboteur"]]
    return {"saboteur_episodes": len(sab), "detection_rate": float(sab["t_flag"].notna().mean()),
            "ttd_mean": float(ttd.mean()), "ttd_std": float(ttd.std()),
            "early_flag_rate": float((sab["t_flag"] < sab["t_first_sabotage"]).mean()),
            "honest_robot_episodes": len(honest), "false_positive_rate": float(honest["t_flag"].notna().mean())}


def monitor_summary(episodes: pd.DataFrame, horizon: int = 500) -> pd.DataFrame:
    """Per (condition, layout): detection rate, TTD (never flagged counts as horizon - t_first_sabotage, early flags as 0), robot-episode FPR."""
    rows = []
    for (cond, layout), g in episodes.groupby(["condition", "layout"], sort=False):
        sab = g[g["t_first_sabotage"].notna()]
        ttd = (sab["t_flag_saboteur"].fillna(horizon) - sab["t_first_sabotage"]).clip(lower=0)
        rows.append({"condition": cond, "layout": layout, "episodes": len(g),
                     "detection_rate": sab["t_flag_saboteur"].notna().mean() if len(sab) else np.nan,
                     "early_flag_rate": (sab["t_flag_saboteur"] < sab["t_first_sabotage"]).mean() if len(sab) else np.nan,
                     "ttd_mean": ttd.mean() if len(sab) else np.nan, "ttd_std": ttd.std() if len(sab) else np.nan,
                     "false_positive_rate": g["false_flags"].sum() / g["honest_robots"].sum()})
    return pd.DataFrame(rows)


def monitor_report(windows: pd.DataFrame, score, tau: float, persist: int = 1) -> dict:
    """Window metrics for both labels plus detection metrics, for one monitor on one set of windows (e.g. the test split)."""
    flagged = np.asarray(score) > tau
    out = {f"{label}_{k}": v for label in ("robot", "window")
           for k, v in window_metrics(windows["is_saboteur" if label == "robot" else "sabotage_window"], score, flagged).items()}
    return {**out, **detection_metrics(first_flag(windows.assign(flag=flagged), persist=persist))}


def paired_difference(episodes: pd.DataFrame, treatment: str, control: str, column: str = "deliveries",
                      resamples: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Per layout: mean of (treatment - control) over episode seeds, with a seeded bootstrap 95% interval.

    ``episodes`` holds one row per episode (condition, layout, episode_seed, column). Seeds are shared by every
    condition, so the two conditions are paired on episode_seed; seeds missing from either condition are dropped.
    """
    rng, rows = np.random.default_rng(seed), []
    for layout, g in episodes.groupby("layout", sort=False):
        pick = lambda c: g[g["condition"] == c].set_index("episode_seed")[column]
        a, b = pick(treatment), pick(control)
        d = (a - b.reindex(a.index)).dropna().to_numpy(float)
        boot = rng.choice(d, (resamples, len(d))).mean(axis=1)
        rows.append({"treatment": treatment, "control": control, "layout": layout, "episodes": len(d),
                     "mean_diff": d.mean(), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "share_improved": float((d > 0).mean())})
    return pd.DataFrame(rows)
