"""Quarantine-and-replan: a flagged robot's tile becomes an obstacle (search with changing costs).

QuarantineMonitor implements the Monitor protocol. It keeps the step rows of the live episode in the log schema,
scores the latest window with the same feature code as the offline dataset (monitor/features.py) and flags a robot
once its score exceeds ``model.tau`` in ``persist`` consecutive windows. Flags are permanent. FleetController then
passes the flagged robots to the honest planners, and blocks each flagged robot's tile at flag time for good.

``model`` is a ThresholdRule or LogisticMonitorModel (anything with ``score(windows)`` and ``tau``). ``debug`` is
set by build_fleet to FleetController.debug_info: robots report their next planned tiles and current target to the
fleet manager, exactly the columns the episode logs hold. Everything else comes from WarehouseState and the actions.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..types import WarehouseState
from .features import window_table


class QuarantineMonitor:
    def __init__(self, model, window: int = 20, stride: int = 10, persist: int = 1,
                 debug: Callable[[], list[dict]] | None = None):
        self.model, self.window, self.stride, self.persist, self.debug = model, window, stride, persist, debug
        self.rows: list[dict] = []
        self.reset(None)

    def reset(self, state: WarehouseState | None) -> None:
        self.rows, self.pending = [], None
        self.highways, self.goals = (state.highways, state.goals) if state else (None, ())
        self._flagged: set[int] = set()
        self._scores: dict[int, float] = {}
        self._streak: dict[int, int] = {}

    def update(self, state: WarehouseState, actions: Sequence[int]) -> None:
        if self.pending is not None:
            self.rows += self._complete(*self.pending, state)
            t_end = state.t - 1  # last step whose row is now complete
            if t_end >= self.window - 1 and (t_end - self.window + 1) % self.stride == 0:
                self._score(t_end)
        debug = self.debug() if self.debug else [{}] * len(state.robots)
        self.pending = (state, list(actions), debug)

    def scores(self) -> dict[int, float]:
        return dict(self._scores)

    def flagged(self) -> frozenset[int]:
        return frozenset(self._flagged)

    def _complete(self, s: WarehouseState, actions, debug, nxt: WarehouseState) -> list[dict]:
        """Log-schema rows for step s.t, using the next state for moved / picked_up / dropped / delivered."""
        requested = set(s.requested_shelf_ids)
        rows = []
        for r, a, d, n in zip(s.robots, actions, debug, nxt.robots):
            carried, was_requested = r.carrying_shelf_id, r.carrying_shelf_id in requested
            target = d.get("target")
            rows.append({
                "t": s.t, "robot_id": r.robot_id, "x": r.x, "y": r.y, "carrying_shelf_id": carried,
                "carrying_requested": was_requested, "on_highway": bool(s.highways[r.y, r.x]),
                "on_chokepoint": (r.x, r.y) in s.chokepoints, "action": a, "moved": (n.x, n.y) != (r.x, r.y),
                "delivered": was_requested and n.carrying_shelf_id == carried and (n.x, n.y) in s.goals,
                "picked_up": carried is None and n.carrying_shelf_id is not None,
                "dropped": carried is not None and n.carrying_shelf_id is None,
                "planned_cells": d.get("planned_cells", []),
                "target_x": np.nan if target is None else target[0],
                "target_y": np.nan if target is None else target[1],
            })
        return rows

    def _score(self, t_end: int) -> None:
        feats = window_table(pd.DataFrame(self.rows), self.highways, self.goals, self.window, self.stride, [t_end])
        for rid, p in zip(feats["robot_id"], self.model.score(feats)):
            self._scores[int(rid)] = float(p)
            self._streak[rid] = self._streak.get(rid, 0) + 1 if p > self.model.tau else 0
            if self._streak[rid] >= self.persist:
                self._flagged.add(int(rid))


def save_monitor(path: Path | str, model, window: int = 20, stride: int = 10, persist: int = 1) -> Path:
    """Store a fitted model with the window settings it was trained for (data/models/ is gitignored)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "window": window, "stride": stride, "persist": persist}, path)
    return path


def load_monitor(path: Path | str) -> QuarantineMonitor:
    """A fresh QuarantineMonitor from a file written by save_monitor."""
    return QuarantineMonitor(**joblib.load(path))
