"""Single-agent view of the warehouse for a learning honest robot (agents and environments; partial observability).

One robot (the learner) is driven by the caller's actions from rware's local observation; the other robots run the
usual fleet (A* honest robots, optionally a scripted saboteur). The A* robots treat the learner as a non-cooperating
robot: it is an obstacle at its current tile and is not in the reservation table, because its moves are not planned.
The saboteur ignores the learner (it targets ``ctx.honest_ids``, the A* robots).

Reward for the learner: the wrapper's honest reward (+10 per delivery, -0.01 per step) plus ``pickup_bonus`` when it
picks up a currently requested shelf (shaping, proposal section on DQN). The bonus is paid once per shelf while it
stays requested; otherwise picking up and dropping the same shelf farms it (seen in the first smoke run).

``learner="astar"`` puts an ordinary A* honest robot in the learner's slot and ignores the given action; it gives the
A* benchmark in exactly the same setting.
"""
from __future__ import annotations

import numpy as np

from ..agents.base import FleetContext
from ..agents.controller import FleetController
from ..agents.honest_astar import HonestAStarPolicy
from ..agents.saboteur_scripted import BEHAVIOURS, ScriptedSaboteur
from ..env.make_env import make_env
from ..env.reward_wrapper import SabotageRewardWrapper
from ..env.state import extract_state
from ..planning.grid_graph import NOOP


class ExternalPolicy:
    """Placeholder in the fleet for the learner; its action is overwritten by the caller."""

    def reset(self, state, robot_id, rng) -> None:
        pass

    def act(self, state) -> int:
        return NOOP

    def debug_info(self) -> dict:
        return {"phase": "learner", "plan_len": np.nan, "plan_ms": np.nan, "nodes_expanded": np.nan,
                "replanned": False, "sabotage_active": False}


class FleetEnv:
    def __init__(self, layout: str, env: dict | None = None, planner: dict | None = None, kind: str = "honest",
                 saboteur: dict | None = None, pickup_bonus: float = 1.0, learner: str = "external"):
        if kind != "honest" and kind not in BEHAVIOURS:
            raise ValueError(f"kind must be 'honest' or a saboteur behaviour, got {kind!r}")
        self.kind, self.planner, self.saboteur_cfg = kind, planner or {}, saboteur or {}
        self.pickup_bonus, self.learner = pickup_bonus, learner
        self.env = SabotageRewardWrapper(make_env({**(env or {}), "layout": layout}), kind in BEHAVIOURS)
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.n_actions = 5

    def reset(self, seed: int) -> np.ndarray:
        obs, info = self.env.reset(seed=seed)
        w = self.env.unwrapped
        n = len(w.agents)
        rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n + 1)]
        sab = info["saboteur_id"]
        self.rid = int(rngs[n].choice([i for i in range(n) if i != sab]))
        astar = {i for i in range(n) if i != sab and (i != self.rid or self.learner == "astar")}
        ctx = FleetContext(self.planner, frozenset(astar), oracle=True)
        policies = []
        for i in range(n):
            if i in astar:
                policies.append(HonestAStarPolicy(ctx))
            elif i == sab:
                policies.append(ScriptedSaboteur(ctx, self.kind, **self.saboteur_cfg.get(self.kind, {}),
                                                 **self.saboteur_cfg.get("common", {})))
            else:
                policies.append(ExternalPolicy())
        self.fleet = FleetController(policies, ctx)
        self.state = extract_state(w, 0)
        self.fleet.reset(self.state, rngs[:n])
        self.sab = sab
        self.bonus_paid: set[int] = set()
        self.stats = {"learner_deliveries": 0, "learner_pickups": 0, "honest_deliveries": 0}
        return np.asarray(obs[self.rid], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Advance every robot one step. Returns (learner observation, learner reward, done)."""
        actions = self.fleet.act(self.state)
        if self.learner == "external":
            actions[self.rid] = int(action)
        requested = set(self.state.requested_shelf_ids)
        obs, rewards, done, _, info = self.env.step(actions)
        self.state = extract_state(self.env.unwrapped, self.state.t + 1)
        ev = info["events"]
        r = float(rewards[self.rid])
        self.bonus_paid &= set(self.state.requested_shelf_ids)
        shelf = self.state.robots[self.rid].carrying_shelf_id
        if ev["picked_up"][self.rid] and shelf in requested:
            self.stats["learner_pickups"] += 1
            if shelf not in self.bonus_paid:
                r += self.pickup_bonus
                self.bonus_paid.add(shelf)
        self.stats["learner_deliveries"] += ev["delivered"][self.rid]
        self.stats["honest_deliveries"] += sum(d for i, d in enumerate(ev["delivered"]) if i != self.sab)
        return np.asarray(obs[self.rid], dtype=np.float32), r, bool(done)

    def close(self) -> None:
        self.env.close()
