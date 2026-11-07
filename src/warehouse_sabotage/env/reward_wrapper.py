"""SabotageRewardWrapper and event detection (reward design for a mixed-motive game).

Honest robot i: +10 per delivery, -0.01 per step.
Saboteur: -10 per honest delivery, +5 when it drops a currently requested shelf at a slot farther (loaded
shortest-path distance) from the nearest goal than where it picked it up, -50 once when first flagged by the monitor.
"""
from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from ..planning.grid_graph import bfs_distances, passable
from .state import extract_state


def loaded_goal_distance(w, tile: tuple[int, int]) -> int:
    """Loaded-robot shortest-path length from ``tile`` to the nearest goal in the current warehouse."""
    mask = passable(extract_state(w, 0), loaded=True)
    mask[tile[1], tile[0]] = True
    return int(bfs_distances(mask, w.goals)[tile[1], tile[0]])


class SabotageRewardWrapper(gym.Wrapper):
    """Draws a per-episode saboteur id from the reset seed, reshapes rewards and adds ``info["events"]``.

    ``flag_hook`` is an optional callable returning the monitor's currently flagged robot ids (None until Week 7).
    """

    def __init__(self, env, has_saboteur: bool, flag_hook: Callable[[], frozenset[int]] | None = None,
                 delivery_reward: float = 10.0, step_penalty: float = 0.01):
        super().__init__(env)
        self.has_saboteur = has_saboteur
        self.flag_hook = flag_hook
        self.delivery_reward = delivery_reward
        self.step_penalty = step_penalty
        self.saboteur_id: int | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        n = len(self.unwrapped.agents)
        rng = np.random.default_rng(None if seed is None else (seed, 1))
        self.saboteur_id = int(rng.integers(n)) if self.has_saboteur else None
        self._flagged = False
        self._pickup_dist: int | None = None
        return obs, {**info, "saboteur_id": self.saboteur_id}

    def step(self, actions):
        w = self.unwrapped
        before = [(a.x, a.y, a.carrying_shelf) for a in w.agents]
        obs, base, done, truncated, info = self.env.step(actions)
        after = [(a.x, a.y, a.carrying_shelf) for a in w.agents]
        n = len(after)

        delivered = [int(r > 0) for r in base]  # INDIVIDUAL reward: +1 to the robot standing on the goal
        moved = [b[:2] != a[:2] for b, a in zip(before, after)]
        picked_up = [b[2] is None and a[2] is not None for b, a in zip(before, after)]
        dropped = [b[2] is not None and a[2] is None for b, a in zip(before, after)]

        rewards = [self.delivery_reward * delivered[i] - self.step_penalty for i in range(n)]
        k = self.saboteur_id
        if k is not None:
            honest_deliveries = sum(d for i, d in enumerate(delivered) if i != k)
            r = -10.0 * honest_deliveries
            if picked_up[k]:
                self._pickup_dist = loaded_goal_distance(w, after[k][:2])
            if dropped[k]:
                shelf = before[k][2]
                if (shelf in w.request_queue and self._pickup_dist is not None
                        and loaded_goal_distance(w, after[k][:2]) > self._pickup_dist):
                    r += 5.0
                self._pickup_dist = None
            if self.flag_hook is not None and not self._flagged and k in self.flag_hook():
                self._flagged = True
                r -= 50.0
            rewards[k] = r

        info = {**info, "saboteur_id": k, "events": {
            "delivered": delivered, "picked_up": picked_up, "dropped": dropped, "moved": moved,
        }}
        return obs, rewards, done, truncated, info
