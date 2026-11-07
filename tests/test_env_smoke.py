"""Smoke test: the stock tiny rware env builds, resets with a seed and steps."""
import gymnasium as gym
import numpy as np
import rware  # noqa: F401  (registers env ids)


def test_tiny_env_smoke():
    env = gym.make("rware-tiny-4ag-v2")
    obs, _ = env.reset(seed=0)
    assert len(obs) == 4
    rng = np.random.default_rng(0)
    for _ in range(50):
        env.step(rng.integers(0, 5, size=4).tolist())
    w = env.unwrapped
    assert w.grid_size == (11, 10)
    assert [tuple(g) for g in w.goals] == [(4, 10), (5, 10)]
    env.close()
