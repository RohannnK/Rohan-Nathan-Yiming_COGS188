"""Double + Dueling DQN honest robot (neural networks as function approximators for Q-learning).

- Dueling head: Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)  (Wang et al. 2016).
- Double DQN target: y = r + gamma * Q_target(s', argmax_a Q_online(s', a))  (van Hasselt et al. 2016).
- Experience replay and a hard-updated target network (Mnih et al. 2015); Huber loss; epsilon-greedy exploration
  with a linear decay over training steps.

The learner sees only rware's local observation (rl/fleet_env.py). Randomness: one numpy Generator per run, which
also seeds torch.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .fleet_env import FleetEnv

log = logging.getLogger(__name__)


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """Fixed-size ring buffer of (s, a, r, s', done)."""

    def __init__(self, capacity: int, obs_dim: int):
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.i = capacity, 0, 0

    def add(self, s, a, r, s2, d) -> None:
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch)
        return tuple(torch.from_numpy(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d))


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, rng: np.random.Generator):
        torch.manual_seed(int(rng.integers(2**31)))
        self.rng, self.n_actions = rng, n_actions
        self.gamma, self.batch = cfg["gamma"], cfg["batch_size"]
        self.double = cfg.get("double", True)
        self.online = DuelingQNet(obs_dim, n_actions, cfg["hidden"])
        self.target = DuelingQNet(obs_dim, n_actions, cfg["hidden"])
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"], obs_dim)

    def act(self, obs: np.ndarray, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.online(torch.from_numpy(obs)[None]).argmax(dim=1).item())

    def update(self) -> float:
        s, a, r, s2, d = self.buffer.sample(self.batch, self.rng)
        with torch.no_grad():
            if self.double:
                a2 = self.online(s2).argmax(dim=1, keepdim=True)
                q2 = self.target(s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.target(s2).max(dim=1).values
            y = r + self.gamma * (1 - d) * q2
        q = self.online(s).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.detach())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())


def linear_eps(step: int, start: float, end: float, decay_steps: int) -> float:
    return start + (end - start) * min(1.0, step / max(1, decay_steps))


def make_fleet_env(cfg: dict, learner: str = "external") -> FleetEnv:
    return FleetEnv(cfg["layout"], cfg.get("env", {}), cfg.get("planner", {}), cfg.get("kind", "honest"),
                    cfg.get("saboteur", {}), cfg["dqn"].get("pickup_bonus", 1.0), learner)


def run_episode(env: FleetEnv, seed: int, policy) -> dict:
    """Play one episode with ``policy(obs) -> action``; returns the learner's return and the delivery counts."""
    obs, done, ret, steps = env.reset(seed), False, 0.0, 0
    while not done:
        obs, r, done = env.step(policy(obs))
        ret, steps = ret + r, steps + 1
    return {"return": ret, "steps": steps, **env.stats}


def evaluate(env: FleetEnv, agent: DQNAgent | None, seeds, eps: float) -> dict:
    """Mean over ``seeds`` of one episode each. ``agent=None`` evaluates whatever drives the learner slot already
    (the A* benchmark when env.learner == 'astar')."""
    policy = (lambda o: 0) if agent is None else (lambda o: agent.act(o, eps))
    rows = pd.DataFrame([run_episode(env, s, policy) for s in seeds])
    return rows.mean(numeric_only=True).to_dict()


def train_dqn(cfg: dict, seed: int, out_dir: str | Path | None = None, smoke: bool = False) -> pd.DataFrame:
    """Train one DQN learner. Returns one row per training episode, with greedy evaluation columns (prefix
    ``eval_``) filled every ``eval_every`` episodes. Saves the final online network to out_dir/seed_<seed>.pt."""
    d = cfg["dqn"]
    total = d["smoke_steps"] if smoke else d["total_steps"]
    rng = np.random.default_rng(seed)
    env, eval_env = make_fleet_env(cfg), make_fleet_env(cfg)
    agent = DQNAgent(env.obs_dim, env.n_actions, d, rng)
    episode_seeds = np.random.default_rng((seed, 1)).integers(10**9, size=total // 100 + 1)
    eval_seeds = cfg["eval_seeds"][: 1 if smoke else None]
    step, ep, rows, t0 = 0, 0, [], time.perf_counter()
    while step < total:
        obs, done, ret, losses = env.reset(int(episode_seeds[ep])), False, 0.0, []
        while not done:
            eps = linear_eps(step, d["eps_start"], d["eps_end"], int(d["eps_decay_frac"] * total))
            a = agent.act(obs, eps)
            obs2, r, done = env.step(a)
            agent.buffer.add(obs, a, r, obs2, 0.0)  # episodes end only at the time limit: never a true terminal
            obs, ret, step = obs2, ret + r, step + 1
            if agent.buffer.size >= d["learning_starts"] and step % d["train_every"] == 0:
                losses.append(agent.update())
            if step % d["target_update"] == 0:
                agent.sync_target()
        row = {"seed": seed, "episode": ep, "step": step, "epsilon": eps, "return": ret,
               "loss": float(np.mean(losses)) if losses else np.nan, "seconds": time.perf_counter() - t0,
               **env.stats}
        if (ep + 1) % d["eval_every"] == 0 or step >= total:
            row.update({f"eval_{k}": v for k, v in evaluate(eval_env, agent, eval_seeds, d["eval_eps"]).items()})
            log.info("seed %d ep %d step %d eps %.2f return %.1f eval deliveries %.2f", seed, ep, step, eps, ret,
                     row["eval_learner_deliveries"])
        rows.append(row)
        ep += 1
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        torch.save(agent.online.state_dict(), Path(out_dir) / f"seed_{seed}.pt")
    env.close()
    eval_env.close()
    return pd.DataFrame(rows)


def dqn_report(cfg: dict, curves: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Summary table of the final evaluation per seed plus two references on the same evaluation seeds: an A* robot
    in the learner's slot and a learner that never moves. Returns (table, A* deliveries, never-moves deliveries)."""
    seeds = cfg["eval_seeds"]
    astar = evaluate(make_fleet_env(cfg, learner="astar"), None, seeds, 0.0)
    noop = evaluate(make_fleet_env(cfg), None, seeds, 0.0)
    rows = []
    for seed, c in curves.groupby("seed"):
        ev = c.dropna(subset=["eval_learner_deliveries"])
        last = ev.iloc[-1]
        rows.append({"policy": "dqn", "seed": seed, "train_steps": int(c["step"].iloc[-1]),
                     "train_seconds": c["seconds"].iloc[-1], "best_eval_learner_deliveries":
                     ev["eval_learner_deliveries"].max(),
                     **{k[5:]: last[k] for k in ev.columns if k.startswith("eval_")}})
    for name, r in (("astar", astar), ("never_moves", noop)):
        rows.append({"policy": name, **r})
    return pd.DataFrame(rows), astar["learner_deliveries"], noop["learner_deliveries"]
