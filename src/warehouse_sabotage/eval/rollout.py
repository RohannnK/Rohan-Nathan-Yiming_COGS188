"""Run one episode and collect per-step logs (agent-environment loop)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..agents.controller import build_fleet
from ..agents.saboteur_scripted import BEHAVIOURS
from ..env.make_env import make_env
from ..env.reward_wrapper import SabotageRewardWrapper
from ..env.state import extract_state


def run_episode(spec: dict, seed: int, on_step: Callable | None = None) -> list[dict]:
    """Run one episode and return one log row per robot per step (project.md section 7 schema).

    ``planned_cells`` holds an honest robot's next (up to 8) planned tiles as [x, y] pairs, starting with the tile
    it should occupy at t+1; it is empty for saboteur, parked and random robots and for honest robots holding still.

    spec keys: run_name, condition, layout, kind ('random', 'honest', 'parked' or a saboteur behaviour),
    env (make_env kwargs), planner, saboteur. ``on_step(env)`` is called after reset and after every step.
    """
    kind = spec["kind"]
    has_saboteur = kind in BEHAVIOURS
    env = SabotageRewardWrapper(make_env({**spec.get("env", {}), "layout": spec["layout"]}), has_saboteur)
    _, info = env.reset(seed=seed)
    w = env.unwrapped
    n = len(w.agents)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n + 1)]
    special = info["saboteur_id"] if has_saboteur else int(rngs[n].integers(n)) if kind == "parked" else None
    fleet = build_fleet(kind, n, special, spec.get("planner", {}), spec.get("saboteur", {}))
    state = extract_state(w, 0)
    fleet.reset(state, rngs[:n])
    if on_step:
        on_step(env)

    behaviour = kind if has_saboteur else "none"
    rows, done = [], False
    while not done:
        actions = fleet.act(state)
        debug = fleet.debug_info()
        _, rewards, done, _, info = env.step(actions)
        ev = info["events"]
        requested = set(state.requested_shelf_ids)
        for r, a, d in zip(state.robots, actions, debug):
            i = r.robot_id
            rows.append({
                "run_name": spec["run_name"], "condition": spec["condition"], "layout": spec["layout"],
                "saboteur_behaviour": behaviour, "episode_seed": seed, "t": state.t, "robot_id": i,
                "is_saboteur": i == special and has_saboteur,
                "x": r.x, "y": r.y, "direction": r.direction, "carrying_shelf_id": r.carrying_shelf_id,
                "carrying_requested": r.carrying_shelf_id in requested,
                "on_highway": bool(state.highways[r.y, r.x]), "on_chokepoint": (r.x, r.y) in state.chokepoints,
                "action": a, "moved": ev["moved"][i], "delivered": bool(ev["delivered"][i]),
                "picked_up": ev["picked_up"][i], "dropped": ev["dropped"][i],
                "sabotage_active": bool(d["sabotage_active"]), "task_phase": d["phase"],
                "plan_len": d["plan_len"], "plan_ms": d["plan_ms"], "nodes_expanded": d["nodes_expanded"],
                "replanned": bool(d["replanned"]), "reward": rewards[i],
                "planned_cells": [list(c) for c in d.get("planned_cells", [])],
            })
        state = extract_state(w, state.t + 1)
        if on_step:
            on_step(env)
    env.close()
    return rows
