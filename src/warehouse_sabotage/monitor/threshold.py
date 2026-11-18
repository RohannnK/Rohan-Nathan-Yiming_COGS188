"""Threshold-rule monitor: flag on chokepoint dwell time or idle ticks (baseline detector).

A window is flagged when choke_steps >= choke_limit or idle_ticks >= idle_limit. The limits are chosen on the
validation split as the pair with the highest recall at a window false-positive rate of at most ``max_fpr``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdRule:
    choke_limit: int
    idle_limit: int
    tau: float = 0.5  # scores are 0 or 1, so any tau in (0, 1) works

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        return ((windows["choke_steps"] >= self.choke_limit) | (windows["idle_ticks"] >= self.idle_limit)).to_numpy(float)


def fit_threshold(windows: pd.DataFrame, window: int = 20, max_fpr: float = 0.05,
                  label: str = "is_saboteur") -> ThresholdRule:
    """Grid search over both limits (0..window) for the best recall with window FPR <= max_fpr."""
    y = windows[label].to_numpy(bool)
    best, best_key = ThresholdRule(window + 1, window + 1), (-1.0, 0.0)
    for c in range(window + 1):
        for i in range(window + 1):
            flagged = ThresholdRule(c, i).score(windows).astype(bool)
            fpr = flagged[~y].mean() if (~y).any() else 0.0
            recall = flagged[y].mean() if y.any() else 0.0
            if fpr <= max_fpr and (recall, -fpr) > best_key:
                best, best_key = ThresholdRule(c, i), (recall, -fpr)
    return best
