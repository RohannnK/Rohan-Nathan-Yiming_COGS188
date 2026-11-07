"""Scripted saboteur behaviours (adversarial agent in an imperfect-information game).

The saboteur is not in the reservation table and plans with plain spatial A*, but it reads the fleet manager's
claims and planned paths (an insider). On each step, with probability ``camouflage``, it instead takes a cover
action that looks like honest work (driving towards a requested shelf), so sabotage is intermittent.

Behaviours:
  goal_blocker     loiters near the delivery area; steps onto the goal a loaded honest robot is heading for when
                   that robot is within ``block_distance`` of a goal.
  corridor_parker  waits beside chokepoint corridors; enters a corridor tile on an honest robot's planned path and
                   parks there for ``park_steps``.
  shelf_hider      picks up a requested shelf and drops it at the empty slot farthest from the goals.
  shelf_squatter   parks underneath a requested shelf that an honest robot has claimed.
"""
from __future__ import annotations

import numpy as np

from ..planning.grid_graph import NOOP, TOGGLE_LOAD, UNREACHABLE, bfs_distances, passable
from ..planning.heuristics import manhattan
from ..types import WarehouseState
from .base import FleetContext, Navigator
from .task_assignment import empty_slots

BEHAVIOURS = ("goal_blocker", "corridor_parker", "shelf_hider", "shelf_squatter")


class ScriptedSaboteur:
    def __init__(self, ctx: FleetContext, behaviour: str, camouflage: float = 0.2, block_distance: int = 5,
                 park_steps: int = 15, lookahead: int = 12, cooldown: int = 20):
        if behaviour not in BEHAVIOURS:
            raise ValueError(f"unknown saboteur behaviour {behaviour!r}")
        self.ctx, self.behaviour, self.camouflage = ctx, behaviour, camouflage
        self.block_distance, self.park_steps = block_distance, park_steps
        self.lookahead, self.cooldown_steps = lookahead, cooldown

    def reset(self, state: WarehouseState, robot_id: int, rng: np.random.Generator) -> None:
        self.rid, self.rng = robot_id, rng
        self.nav = Navigator(robot_id, self.ctx, cooperative=False)
        self.phase, self.active = "start", False
        self.park_left = self.cooldown = 0
        self.loiter_target = self.drop_target = None
        goals = set(state.goals)
        self.goal_area = {c for c in state.chokepoints if min(manhattan(c, g) for g in goals) <= 1}
        self.corridor = set(state.chokepoints) - self.goal_area
        h, w = state.highways.shape
        highway = [(x, y) for y in range(h) for x in range(w) if state.highways[y, x]]
        self.loiter_tiles = [c for c in highway if c not in state.chokepoints
                             and 2 <= min(manhattan(c, g) for g in goals) <= 4]
        self.wait_tiles = [c for c in highway if c not in state.chokepoints
                           and any(manhattan(c, k) == 1 for k in self.corridor)]

    def act(self, state: WarehouseState) -> int:
        self.nav.begin_step()
        me = state.robots[self.rid]
        if me.carrying_shelf_id is None and self.rng.random() < self.camouflage:
            return self._cover(state)
        return getattr(self, "_" + self.behaviour)(state, me)

    def debug_info(self) -> dict:
        return {"phase": self.phase, **self.nav.debug(), "sabotage_active": self.active}

    # helpers ---------------------------------------------------------------------------------------------------
    def _go(self, state, target, phase, active, loaded=False, avoid=frozenset(), at_target=NOOP) -> int:
        self.phase, self.active = phase, active
        a = self.nav.step(state, target, loaded, avoid)
        return at_target if a is None else a

    def _cover(self, state) -> int:
        """Honest-looking step: drive towards the nearest requested shelf without picking it up."""
        me = state.robots[self.rid]
        targets = [state.shelf_positions[s] for s in state.requested_shelf_ids]
        target = min(targets, key=lambda p: (manhattan((me.x, me.y), p), p))
        return self._go(state, target, "cover", False)

    def _honest(self, state):
        return [r for r in state.robots if r.robot_id in self.ctx.honest_ids]

    # behaviours ------------------------------------------------------------------------------------------------
    def _goal_blocker(self, state, me) -> int:
        req = set(state.requested_shelf_ids)
        threats = [r for r in self._honest(state) if r.carrying_shelf_id in req
                   and min(manhattan((r.x, r.y), g) for g in state.goals) <= self.block_distance]
        if threats:
            r = min(threats, key=lambda r: min(manhattan((r.x, r.y), g) for g in state.goals))
            path = set(self.ctx.plans.get(r.robot_id, []))
            if (me.x, me.y) in self.goal_area and (me.x, me.y) in path:
                return self._go(state, (me.x, me.y), "block", True)
            target = self.ctx.board.tile.get(r.robot_id)
            if target not in state.goals:
                target = min(state.goals, key=lambda g: manhattan((r.x, r.y), g))
            return self._go(state, target, "block", True)
        if self.loiter_target is None or (me.x, me.y) == self.loiter_target:
            self.loiter_target = self.loiter_tiles[self.rng.integers(len(self.loiter_tiles))]
        return self._go(state, self.loiter_target, "loiter", False)

    def _corridor_parker(self, state, me) -> int:
        if self.park_left > 0 and (me.x, me.y) in self.corridor:
            self.park_left -= 1
            return self._go(state, (me.x, me.y), "park", True)
        used = [c for rid in self.ctx.honest_ids for c in self.ctx.plans.get(rid, [])[: self.lookahead]
                if c in self.corridor]
        if used:
            target = min(used, key=lambda c: (manhattan((me.x, me.y), c), c))
            a = self._go(state, target, "enter", True)
            if (me.x, me.y) == target:
                self.park_left = self.park_steps
            return a
        target = min(self.wait_tiles, key=lambda c: (manhattan((me.x, me.y), c), c))
        return self._go(state, target, "wait", False)

    def _shelf_hider(self, state, me) -> int:
        goals = frozenset(state.goals)  # never step on a goal while carrying: that would deliver the shelf
        if me.carrying_shelf_id is not None:
            hiding = me.carrying_shelf_id in state.requested_shelf_ids
            if self.drop_target is None or self.drop_target not in empty_slots(state):
                self.drop_target = self._farthest_slot(state, me) if hiding else self._nearest_slot(state, me)
            target = self.drop_target
            if (me.x, me.y) == target:
                self.drop_target, self.cooldown = None, self.cooldown_steps
            return self._go(state, target, "hide" if hiding else "return", hiding, loaded=True,
                            avoid=goals, at_target=TOGGLE_LOAD)
        if self.cooldown > 0:
            self.cooldown -= 1
            return self._cover(state)
        carried = {r.carrying_shelf_id for r in state.robots}
        options = [state.shelf_positions[s] for s in state.requested_shelf_ids if s not in carried]
        if not options:
            return self._cover(state)
        target = min(options, key=lambda p: (manhattan((me.x, me.y), p), p))
        a = self._go(state, target, "seek", False, at_target=TOGGLE_LOAD)
        if a == TOGGLE_LOAD:
            self.active = True  # the pick-up itself is the sabotage manoeuvre
        return a

    def _shelf_squatter(self, state, me) -> int:
        carried = {r.carrying_shelf_id for r in state.robots}
        req = set(state.requested_shelf_ids)
        claimed = [state.shelf_positions[s] for r, s in self.ctx.board.shelf.items()
                   if s in req and s not in carried and r in self.ctx.honest_ids]
        if not claimed:
            return self._cover(state)
        target = min(claimed, key=lambda p: (manhattan((me.x, me.y), p), p))
        return self._go(state, target, "squat", True)

    def _slot_distances(self, state, me):
        mask = passable(state, loaded=True)
        slots = [s for s in empty_slots(state) if s not in {(r.x, r.y) for r in state.robots if r is not me}]
        return mask, slots, bfs_distances(mask, [(me.x, me.y)])

    def _farthest_slot(self, state, me):
        mask, slots, from_me = self._slot_distances(state, me)
        from_goals = bfs_distances(mask, state.goals)
        reachable = [s for s in slots if from_me[s[1], s[0]] < UNREACHABLE] or [(me.x, me.y)]
        return max(reachable, key=lambda s: (from_goals[s[1], s[0]] % UNREACHABLE, s))

    def _nearest_slot(self, state, me):
        _, slots, from_me = self._slot_distances(state, me)
        return min(slots, key=lambda s: (from_me[s[1], s[0]], s)) if slots else (me.x, me.y)
