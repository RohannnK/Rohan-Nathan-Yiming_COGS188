"""Random and parked idle policies (baseline agents)."""
from __future__ import annotations

import numpy as np

from ..planning.grid_graph import NOOP
from ..types import WarehouseState
from .base import FleetContext, Navigator
from .task_assignment import parking_tiles

_NO_PLAN = {"plan_len": np.nan, "plan_ms": np.nan, "nodes_expanded": np.nan, "replanned": False}


class RandomPolicy:
    """Uniformly random rware actions: the lower baseline."""

    def reset(self, state: WarehouseState, robot_id: int, rng: np.random.Generator) -> None:
        self.rng = rng

    def act(self, state: WarehouseState) -> int:
        return int(self.rng.integers(5))

    def debug_info(self) -> dict:
        return {"phase": "random", **_NO_PLAN, "sabotage_active": False}


class ParkedPolicy:
    """Control robot for the idle condition: drives under a non-requested shelf and stays there."""

    def __init__(self, ctx: FleetContext):
        self.ctx = ctx

    def reset(self, state: WarehouseState, robot_id: int, rng: np.random.Generator) -> None:
        self.rid = robot_id
        self.nav = Navigator(robot_id, self.ctx, cooperative=False)
        self.target = None

    def act(self, state: WarehouseState) -> int:
        self.nav.begin_step()
        me = state.robots[self.rid]
        options = parking_tiles(state)
        if (me.x, me.y) in options:
            self.target = (me.x, me.y)
            return NOOP
        if self.target not in options:
            self.target = min(options, key=lambda p: abs(p[0] - me.x) + abs(p[1] - me.y))
        a = self.nav.step(state, self.target, loaded=False)
        return NOOP if a is None else a

    def debug_info(self) -> dict:
        return {"phase": "parked", **self.nav.debug(), "sabotage_active": False}
