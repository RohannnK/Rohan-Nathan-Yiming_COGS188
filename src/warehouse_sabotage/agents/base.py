"""Policy protocol helpers (rational agent abstraction): shared fleet context and a plan-following navigator.

The Navigator follows an A* plan to a target and replans when a trigger fires (project.md section 9): no plan,
target changed, move failed, path blocked by a non-honest robot, reservation conflict, or every K steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..planning.cooperative import ReservationTable, space_time_astar
from ..planning.grid_graph import NOOP, passable
from ..planning.search import search
from ..types import WarehouseState
from .task_assignment import TaskBoard


@dataclass
class FleetContext:
    """State shared by the fleet manager's policies (honest robots are centrally planned)."""
    planner: dict
    honest_ids: frozenset[int]
    table: ReservationTable = field(default_factory=ReservationTable)
    board: TaskBoard = field(default_factory=TaskBoard)
    plans: dict = field(default_factory=dict)  # honest robot -> remaining planned cells
    flagged: frozenset[int] = frozenset()  # monitor output (Week 7); treated as obstacles


class Navigator:
    """Moves one robot towards a target cell, replanning on triggers and logging planner diagnostics."""

    def __init__(self, rid: int, ctx: FleetContext, cooperative: bool):
        self.rid, self.ctx, self.cooperative = rid, ctx, cooperative
        self.horizon = ctx.planner.get("horizon", 32)
        self.replan_every = ctx.planner.get("replan_every", 8)
        self.reset()

    def reset(self):
        self.states: list[tuple] = []
        self.actions: list[int] = []
        self.target = None
        self.expected = None
        self.last_plan_t = -(10**9)
        self.move_failed = False
        self.begin_step()

    def begin_step(self):
        self.plan_ms = math.nan
        self.nodes = math.nan
        self.replanned = False

    def sync(self, state: WarehouseState):
        """Detect a failed move before anyone acts, and stop advertising a stale plan."""
        me = state.robots[self.rid]
        pos = (me.x, me.y, me.direction)
        self.move_failed = self.expected is not None and pos != self.expected
        if self.move_failed:
            self.states, self.actions = [], []
            if self.cooperative:
                self.hold(state)

    def obstacles(self, state: WarehouseState, target) -> frozenset:
        others = [r for r in state.robots if r.robot_id != self.rid
                  and (not self.cooperative or r.robot_id not in self.ctx.honest_ids or r.robot_id in self.ctx.flagged)]
        return frozenset((r.x, r.y) for r in others) - {tuple(target)}

    def hold(self, state: WarehouseState):
        """Stay put: reserve the current cell for the next few steps and drop the plan."""
        me = state.robots[self.rid]
        self.states, self.actions, self.expected = [], [], (me.x, me.y, me.direction)
        if self.cooperative:
            self.ctx.table.clear(self.rid)
            self.ctx.table.reserve_path(self.rid, state.t, [(me.x, me.y)], stay=2)
            self.ctx.plans[self.rid] = []

    def step(self, state: WarehouseState, target, loaded: bool, avoid=frozenset()) -> int | None:
        """Next action towards target, or None if already there (caller decides what to do)."""
        me = state.robots[self.rid]
        pos = (me.x, me.y, me.direction)
        target = tuple(target)
        if pos[:2] == target:
            self.hold(state)
            return None
        blocked = self.obstacles(state, target) | (frozenset(avoid) - {target})
        t = state.t
        table = self.ctx.table
        need = (
            not self.actions
            or target != self.target
            or self.move_failed
            or any(s[:2] in blocked for s in self.states)
            or t - self.last_plan_t >= self.replan_every
            or (self.cooperative and not table.move_ok(self.rid, t, pos[:2], self.states[0][:2]))
        )
        if need:
            self._plan(state, pos, target, loaded, blocked)
        if not self.actions:  # no path: wait
            self.hold(state)
            return NOOP
        self.expected = self.states.pop(0)
        a = self.actions.pop(0)
        if self.cooperative:
            self.ctx.plans[self.rid] = [s[:2] for s in self.states]
        return a

    def _plan(self, state, pos, target, loaded, blocked):
        mask = passable(state, loaded)
        mask[target[1], target[0]] = True
        if self.cooperative:
            self.ctx.table.clear(self.rid)
            r = space_time_astar(pos, state.t, target, mask, blocked, self.ctx.table, self.rid, self.horizon)
            cells = [s[:2] for s in r.path] if r.found else [pos[:2]]
            self.ctx.table.reserve_path(self.rid, state.t, cells, stay=1 if r.found else 2)
        else:
            r = search(pos, target, mask, "astar", blocked)
        self.replanned = target == self.target  # same target: a trigger fired, not a new task
        self.states, self.actions = r.path[1:], r.actions
        self.target = target
        self.last_plan_t = state.t
        self.plan_ms, self.nodes = r.ms, r.nodes_expanded

    def debug(self) -> dict:
        return {"plan_len": len(self.actions), "plan_ms": self.plan_ms,
                "nodes_expanded": self.nodes, "replanned": self.replanned}
