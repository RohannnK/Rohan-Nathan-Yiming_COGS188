"""Honest robot: task state machine with A* replanning (informed search in a dynamic world).

idle -> claim a requested shelf -> to_shelf (unloaded) -> TOGGLE_LOAD -> to_goal (loaded, nearest free goal)
-> deliver -> return_shelf (nearest empty slot) -> TOGGLE_LOAD -> idle. Idle robots park under a non-requested
shelf, out of loaded robots' way. A stuck counter triggers a randomised yield to break honest-honest deadlocks.
"""
from __future__ import annotations

import numpy as np

from ..planning.grid_graph import DELTA, NOOP, TOGGLE_LOAD, passable
from ..types import WarehouseState
from .base import FleetContext, Navigator
from .task_assignment import empty_slots, parking_tiles


PLANNED_CELLS_LOGGED = 8  # next tiles logged for Week 7 aisle-blocking features


class HonestAStarPolicy:
    def __init__(self, ctx: FleetContext):
        self.ctx = ctx
        self.stuck_limit = ctx.planner.get("stuck_limit", 6)
        self.yield_prob = ctx.planner.get("yield_prob", 0.5)

    def reset(self, state: WarehouseState, robot_id: int, rng: np.random.Generator) -> None:
        self.rid, self.rng = robot_id, rng
        self.nav = Navigator(robot_id, self.ctx, cooperative=True)
        self.phase = "idle"
        self.stuck = 0
        self.prev_pos = None
        self.yield_left = 0
        self.yield_target = None

    def sync(self, state: WarehouseState) -> None:
        self.nav.sync(state)

    def _others(self, state, honest: bool):
        return {(r.x, r.y) for r in state.robots
                if r.robot_id != self.rid and (r.robot_id in self.ctx.honest_ids) == honest}

    def act(self, state: WarehouseState) -> int:
        self.nav.begin_step()
        me = state.robots[self.rid]
        pos = (me.x, me.y)
        self.stuck = self.stuck + 1 if pos == self.prev_pos and self.phase != "idle" else 0
        self.prev_pos = pos
        loaded = me.carrying_shelf_id is not None

        if self.yield_left:
            return self._continue_yield(state, loaded)
        if self.stuck >= self.stuck_limit and self.rng.random() < self.yield_prob:
            if self._start_yield(state, loaded):
                return self._continue_yield(state, loaded)

        board = self.ctx.board
        rogue_tiles = self._others(state, honest=False)
        target, toggle_at_target = self._choose_target(state, me, loaded, rogue_tiles)
        if target is None:
            self.nav.hold(state)
            return NOOP
        a = self.nav.step(state, target, loaded)
        if a is None:  # at target
            if toggle_at_target:
                board.tile.pop(self.rid, None)
                return TOGGLE_LOAD
            return NOOP
        return a

    def _choose_target(self, state, me, loaded, rogue_tiles):
        board, rid = self.ctx.board, self.rid
        req = state.requested_shelf_ids
        old_phase = self.phase
        if loaded:
            board.shelf.pop(rid, None)
            if me.carrying_shelf_id in req:
                self.phase = "to_goal"
                options = [g for g in state.goals if g not in rogue_tiles] or list(state.goals)
                toggle = False
            else:
                self.phase = "return_shelf"
                options = empty_slots(state)
                toggle = True
        else:
            sid = board.shelf.get(rid)
            carried = {r.carrying_shelf_id for r in state.robots}
            if (sid is None or sid not in req or sid in carried
                    or (self.stuck and state.shelf_positions[sid] in rogue_tiles)):
                sid = board.claim_shelf(rid, state, avoid_tiles=rogue_tiles)
            if sid is not None:
                if self.phase != "to_shelf":
                    board.tile.pop(rid, None)
                self.phase = "to_shelf"
                return state.shelf_positions[sid], True
            self.phase = "idle"
            options = parking_tiles(state)
            toggle = False
            if (me.x, me.y) in options:
                board.tile[rid] = (me.x, me.y)
                return (me.x, me.y), False
        if self.phase != old_phase:
            board.tile.pop(rid, None)
        target = board.tile.get(rid)
        if target not in options:
            target = board.claim_tile(rid, state, options, loaded)
        return target, toggle

    def _start_yield(self, state, loaded) -> bool:
        """Randomised yield: step aside to a random free neighbour, then wait 0-2 steps."""
        me = state.robots[self.rid]
        mask = passable(state, loaded)
        occupied = {(r.x, r.y) for r in state.robots}
        h, w = mask.shape
        options = [(me.x + dx, me.y + dy) for dx, dy in DELTA.values()
                   if 0 <= me.x + dx < w and 0 <= me.y + dy < h and mask[me.y + dy, me.x + dx]
                   and (me.x + dx, me.y + dy) not in occupied]
        self.stuck = 0
        if not options:
            return False
        self.yield_target = options[self.rng.integers(len(options))]
        self.yield_left = 3 + int(self.rng.integers(3))  # turn(s) + move + 0-2 waits
        return True

    def _continue_yield(self, state, loaded) -> int:
        self.yield_left -= 1
        a = self.nav.step(state, self.yield_target, loaded)
        return NOOP if a is None else a

    def debug_info(self) -> dict:
        return {"phase": self.phase, **self.nav.debug(), "sabotage_active": False,
                "planned_cells": self.nav.planned[:PLANNED_CELLS_LOGGED]}
