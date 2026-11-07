"""Build the rware env from a config dict (agents and environments)."""
from __future__ import annotations

import gymnasium as gym
import rware  # noqa: F401  (registers env ids)

from .layouts import LAYOUTS, chokepoints


def make_env(cfg: dict) -> gym.Env:
    """cfg keys: layout (name in LAYOUTS), n_agents (default 4), max_steps (default 500), sensor_range (default 1).

    The chosen layout's chokepoints are stored on ``env.unwrapped.chokepoints``.
    """
    n = cfg.get("n_agents", 4)
    env = gym.make(
        f"rware-tiny-{n}ag-v2",
        layout=LAYOUTS[cfg.get("layout", "tiny")],
        max_steps=cfg.get("max_steps", 500),
        sensor_range=cfg.get("sensor_range", 1),
        disable_env_checker=True,
    )
    w = env.unwrapped
    w.chokepoints = chokepoints(w.highways, w.goals)
    return env
