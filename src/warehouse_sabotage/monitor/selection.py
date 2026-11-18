"""Episode-level choice of the flagging setting (Version 2): replay windows in time order, apply the persistence rule.

Version 1 chose the threshold so that the window false-positive rate was at most 5%. One honest robot has about 48
windows per episode, so that left roughly six honest robots in ten flagged at least once. Here every candidate setting
(a threshold, and how many consecutive flagged windows are needed) is replayed over each robot's windows in time
order, exactly as QuarantineMonitor does online, and judged by the robot-episode false-positive rate.

Among the settings that stay within ``max_fpr`` the one with the highest detection rate wins (then the lowest TTD).
A setting that never detects has no false positives, so ``useful`` reports whether the winner detects at least
``min_detection`` of the saboteurs (a diagnostic, not a project requirement). If no candidate stays within ``max_fpr`` at
all, the constraint is not relaxed: the candidate with the lowest false-positive rate is returned, ``useful`` is False
and the caller must report the failure. ``diagnose`` summarises the trade-off for that report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.metrics import detection_metrics

KEYS = ["run_name", "episode_seed", "robot_id"]


class Replay:
    """Windows sorted once by robot-episode and time; ``flags`` replays one candidate (same output as eval.metrics.first_flag)."""

    def __init__(self, windows: pd.DataFrame):
        w = windows.reset_index(drop=True).sort_values(KEYS + ["t_end"], kind="stable")
        self.order = w.index.to_numpy()
        w = w.reset_index(drop=True)
        gid = w.groupby(KEYS, sort=False).ngroup().to_numpy()
        pos = np.arange(len(w))
        first = np.r_[True, gid[1:] != gid[:-1]]
        self.pos, self.gid, self.t_end = pos, gid, w["t_end"].to_numpy()
        self.start = np.maximum.accumulate(np.where(first, pos, 0))
        self.meta = (w.groupby(KEYS, sort=False)
                     .agg(is_saboteur=("is_saboteur", "first"), t_first_sabotage=("t_first_sabotage", "first"),
                          horizon=("horizon", "first")).reset_index())

    def flags(self, flag, persist: int = 1) -> pd.DataFrame:
        """Per robot-episode: first window end with ``persist`` consecutive flagged windows (NaN if never)."""
        f = np.asarray(flag, bool)[self.order]
        streak = self.pos - np.maximum.accumulate(np.where(f, self.start - 1, self.pos))  # run of flagged windows
        hit = streak >= persist
        t_flag = pd.Series(self.t_end[hit]).groupby(self.gid[hit]).min()
        return self.meta.assign(t_flag=t_flag.reindex(range(len(self.meta))).to_numpy(float))


def select_flagging(windows: pd.DataFrame, candidates, max_fpr: float = 0.05, persists=(1, 2, 3, 4, 6),
                    min_detection: float = 0.5):
    """candidates: (setting, flag per window) pairs. Returns (setting, persist, table of every trial, useful)."""
    replay, rows = Replay(windows), []
    for setting, flag in candidates:
        for persist in persists:
            rows.append({"setting": setting, "persist": persist, **detection_metrics(replay.flags(flag, persist))})
    table = pd.DataFrame(rows)
    ok = table[table["false_positive_rate"] <= max_fpr]
    if len(ok):
        best = ok.sort_values(["detection_rate", "ttd_mean"], ascending=[False, True], kind="stable").iloc[0]
    else:  # nothing meets the constraint: report the closest candidate, never pretend it is feasible
        best = table.sort_values(["false_positive_rate", "detection_rate"], ascending=[True, False], kind="stable").iloc[0]
    return best["setting"], int(best["persist"]), table, bool(len(ok) and best["detection_rate"] >= min_detection)


def diagnose(table: pd.DataFrame, max_fpr: float = 0.05) -> dict:
    """Trade-off summary of a ``select_flagging`` table: how many settings, the lowest FPR reachable, the best detection
    within ``max_fpr`` (NaN if none) and the setting with the best detection-minus-FPR balance, constraint ignored."""
    ok = table[table["false_positive_rate"] <= max_fpr]
    balance = table.loc[(table["detection_rate"] - table["false_positive_rate"]).idxmax()]
    return {"candidates": len(table), "min_fpr": float(table["false_positive_rate"].min()),
            "feasible": len(ok), "best_detection_within_fpr": float(ok["detection_rate"].max()) if len(ok) else np.nan,
            "balanced_setting": str(balance["setting"]), "balanced_persist": int(balance["persist"]),
            "balanced_detection": float(balance["detection_rate"]), "balanced_fpr": float(balance["false_positive_rate"])}
