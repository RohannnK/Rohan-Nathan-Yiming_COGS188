"""Single-aisle gridworld MDP with a stochastic blocker (MDPs; Sutton and Barto's cliff walking as the template).

A robot goes from S to G. The short route runs along a 1-wide aisle (`a`) next to the blocker's lane (`b`); the long
route is a detour along the top row. Shelves (`x`) and walls are impassable: a move into them leaves the robot in
place.

- Moving into the lane is a collision with the blocker.
- Ending a step on an aisle tile risks a collision with probability ``bump_prob``: the blocker steps out of its lane.

A collision costs ``collision_reward`` (on top of the step reward) and sends the robot back to S; it is the
deadlock event of this experiment. Reaching G ends the episode. The blocker is memoryless, so the state is just the
robot's tile and the transition model can be written out exactly for value iteration (rl/dp.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LAYOUT = (
    "..........",
    ".xxxxxxxx.",
    ".aaaaaaaa.",
    "SbbbbbbbbG",
)
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))  # UP, RIGHT, DOWN, LEFT as (dx, dy)


@dataclass
class AisleGridworld:
    layout: tuple[str, ...] = LAYOUT
    step_reward: float = -1.0
    collision_reward: float = -20.0
    bump_prob: float = 0.01
    max_steps: int = 200
    # Transition model: P[s, a] has two outcomes (no collision, collision) with these arrays, shape (S, A, 2).
    next_s: np.ndarray = field(init=False, repr=False)
    prob: np.ndarray = field(init=False, repr=False)
    reward: np.ndarray = field(init=False, repr=False)
    collided: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.layout = tuple(self.layout)
        self.h, self.w = len(self.layout), len(self.layout[0])
        cells = {(x, y): c for y, row in enumerate(self.layout) for x, c in enumerate(row)}
        self.start = self.index(*next(p for p, c in cells.items() if c == "S"))
        self.goal = self.index(*next(p for p, c in cells.items() if c == "G"))
        n, A = self.h * self.w, len(MOVES)
        self.n_states, self.n_actions = n, A
        self.next_s = np.zeros((n, A, 2), dtype=int)
        self.prob = np.zeros((n, A, 2))
        self.reward = np.zeros((n, A, 2))
        self.collided = np.zeros((n, A, 2), dtype=bool)
        hit = self.step_reward + self.collision_reward
        for (x, y), c in cells.items():
            s = self.index(x, y)
            for a, (dx, dy) in enumerate(MOVES):
                nx, ny = x + dx, y + dy
                target = cells.get((nx, ny), "x")
                if s == self.goal:  # absorbing, never visited during learning
                    outcome = [(s, 1.0, 0.0, False), (s, 0.0, 0.0, True)]
                elif target == "b":
                    outcome = [(s, 0.0, 0.0, False), (self.start, 1.0, hit, True)]
                elif target == "x":
                    stay_p = self.bump_prob if c == "a" else 0.0
                    outcome = [(s, 1 - stay_p, self.step_reward, False), (self.start, stay_p, hit, True)]
                else:
                    t = self.index(nx, ny)
                    p = self.bump_prob if target == "a" else 0.0
                    outcome = [(t, 1 - p, self.step_reward, False), (self.start, p, hit, True)]
                for k, (s2, p, r, col) in enumerate(outcome):
                    self.next_s[s, a, k], self.prob[s, a, k] = s2, p
                    self.reward[s, a, k], self.collided[s, a, k] = r, col

    def index(self, x: int, y: int) -> int:
        return y * self.w + x

    def xy(self, s: int) -> tuple[int, int]:
        return s % self.w, s // self.w

    def step(self, s: int, a: int, rng: np.random.Generator) -> tuple[int, float, bool, bool]:
        """Sample one transition: (next state, reward, collided, reached goal)."""
        k = int(rng.random() < self.prob[s, a, 1])
        s2 = int(self.next_s[s, a, k])
        return s2, float(self.reward[s, a, k]), bool(self.collided[s, a, k]), s2 == self.goal

    def greedy_path(self, policy: np.ndarray, max_len: int = 100) -> list[tuple[int, int]]:
        """Tiles visited by a deterministic policy from S, assuming no collision (for drawing learned paths)."""
        s, path = self.start, [self.xy(self.start)]
        while s != self.goal and len(path) <= max_len:
            s = int(self.next_s[s, policy[s], 0]) if self.prob[s, policy[s], 0] > 0 else self.start
            path.append(self.xy(s))
        return path

    def route(self, policy: np.ndarray) -> str:
        """'aisle', 'detour' or 'none' (does not reach G) for a deterministic policy."""
        path = self.greedy_path(policy)
        if path[-1] != self.xy(self.goal):
            return "none"
        rows = {y for _, y in path}
        return "aisle" if 2 in rows and 0 not in rows else "detour" if 0 in rows else "other"
