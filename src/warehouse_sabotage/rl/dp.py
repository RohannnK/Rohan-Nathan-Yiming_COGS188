"""Value iteration and policy evaluation on the single-aisle gridworld (dynamic programming, Bellman equations).

Value iteration gives the exact optimal policy that SARSA and Q-learning are compared against; policy evaluation
scores a learned greedy policy exactly instead of by noisy rollouts.
"""
from __future__ import annotations

import numpy as np

from .gridworld import AisleGridworld


def q_from_v(env: AisleGridworld, v: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s, a) = sum over outcomes of p * (r + gamma * V(s')), with V(goal) = 0."""
    v = v.copy()
    v[env.goal] = 0.0
    return (env.prob * (env.reward + gamma * v[env.next_s])).sum(axis=2)


def value_iteration(env: AisleGridworld, gamma: float, tol: float = 1e-8,
                    max_iter: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (V*, greedy policy). Stops when the largest Bellman update is below ``tol``."""
    v = np.zeros(env.n_states)
    for _ in range(max_iter):
        new = q_from_v(env, v, gamma).max(axis=1)
        new[env.goal] = 0.0
        if np.abs(new - v).max() < tol:
            v = new
            break
        v = new
    return v, q_from_v(env, v, gamma).argmax(axis=1)


def policy_evaluation(env: AisleGridworld, policy: np.ndarray, gamma: float, tol: float = 1e-8,
                      max_iter: int = 10_000) -> np.ndarray:
    """Exact V^pi for a deterministic policy (iterative; a policy that never reaches G diverges to a large cost)."""
    v = np.zeros(env.n_states)
    idx = np.arange(env.n_states)
    for _ in range(max_iter):
        new = q_from_v(env, v, gamma)[idx, policy]
        new[env.goal] = 0.0
        if np.abs(new - v).max() < tol:
            return new
        v = new
    return v
