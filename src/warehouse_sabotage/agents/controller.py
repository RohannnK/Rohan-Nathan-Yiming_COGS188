"""FleetController: state to joint action (multi-agent systems).

Owns one Policy per robot, the shared fleet context (reservation table, claims, plans) and an optional Monitor
whose flagged robots are passed to honest planners as obstacles.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import Monitor, Policy, WarehouseState
from .base import FleetContext
from .honest_astar import HonestAStarPolicy
from .saboteur_scripted import BEHAVIOURS, ScriptedSaboteur
from .simple import ParkedPolicy, RandomPolicy


class FleetController:
    def __init__(self, policies: Sequence[Policy], ctx: FleetContext, monitor: Monitor | None = None):
        self.policies, self.ctx, self.monitor = list(policies), ctx, monitor

    def reset(self, state: WarehouseState, rngs: Sequence[np.random.Generator]) -> None:
        for i, p in enumerate(self.policies):
            p.reset(state, i, rngs[i])
        if self.monitor:
            self.monitor.reset(state)

    def act(self, state: WarehouseState) -> list[int]:
        if self.monitor:
            self.ctx.flagged = self.monitor.flagged()
        for p in self.policies:  # detect failed moves before anyone plans against stale reservations
            if hasattr(p, "sync"):
                p.sync(state)
        actions = [int(p.act(state)) for p in self.policies]
        if self.monitor:
            self.monitor.update(state, actions)
        return actions

    def debug_info(self) -> list[dict]:
        return [p.debug_info() for p in self.policies]


def build_fleet(kind: str, n_agents: int, special_id: int | None, planner: dict, saboteur: dict) -> FleetController:
    """kind: 'random' (all random), 'honest' (all honest), 'parked' (special robot parked) or a saboteur behaviour
    (special robot runs it). Every other robot runs HonestAStarPolicy."""
    if kind == "random":
        return FleetController([RandomPolicy() for _ in range(n_agents)], FleetContext(planner, frozenset()))
    honest = frozenset(i for i in range(n_agents) if kind == "honest" or i != special_id)
    ctx = FleetContext(planner, honest)
    policies = []
    for i in range(n_agents):
        if i in honest:
            policies.append(HonestAStarPolicy(ctx))
        elif kind == "parked":
            policies.append(ParkedPolicy(ctx))
        elif kind in BEHAVIOURS:
            policies.append(ScriptedSaboteur(ctx, kind, **saboteur.get(kind, {}), **saboteur.get("common", {})))
        else:
            raise ValueError(f"unknown fleet kind {kind!r}")
    return FleetController(policies, ctx)
