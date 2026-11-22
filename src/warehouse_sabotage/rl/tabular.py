"""SARSA vs. Q-learning on the single-aisle gridworld (TD learning; on-policy vs. off-policy control; epsilon-greedy
exploration as in the bandit lectures).

SARSA:      Q(s,a) += alpha * [r + gamma * Q(s', a') - Q(s,a)]      (a' is the action it will actually take)
Q-learning: Q(s,a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s,a)]

The hypothesis (proposal, after cliff walking): with constant epsilon, Q-learning learns the short aisle next to the
blocker and collides more while exploring; SARSA learns the safer detour.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .dp import policy_evaluation, value_iteration
from .gridworld import AisleGridworld

METHODS = ("sarsa", "q_learning")


def epsilon(schedule: dict, episode: int, n_episodes: int) -> float:
    """'constant': eps. 'linear': start -> end over the first `frac` of training, then end."""
    if schedule["kind"] == "constant":
        return schedule["eps"]
    if schedule["kind"] == "linear":
        span = max(1, int(schedule["frac"] * n_episodes))
        return schedule["start"] + (schedule["end"] - schedule["start"]) * min(1.0, episode / span)
    raise ValueError(f"unknown epsilon schedule {schedule['kind']!r}")


def eps_greedy(q_row: np.ndarray, eps: float, rng: np.random.Generator) -> int:
    """Random action with probability eps, else greedy with random tie-breaking."""
    if rng.random() < eps:
        return int(rng.integers(len(q_row)))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[rng.integers(len(best))])


def train(env: AisleGridworld, method: str, episodes: int, alpha: float, gamma: float, schedule: dict,
          rng: np.random.Generator) -> tuple[np.ndarray, pd.DataFrame]:
    """Train one agent. Returns (Q, per-episode log with return, collisions, steps, epsilon)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    q = np.zeros((env.n_states, env.n_actions))
    log = []
    for ep in range(episodes):
        eps = epsilon(schedule, ep, episodes)
        s, ret, hits, steps, done = env.start, 0.0, 0, 0, False
        a = eps_greedy(q[s], eps, rng)
        while not done and steps < env.max_steps:
            s2, r, hit, done = env.step(s, a, rng)
            a2 = eps_greedy(q[s2], eps, rng)
            if done:
                target = r
            elif method == "sarsa":
                target = r + gamma * q[s2, a2]
            else:
                target = r + gamma * q[s2].max()
            q[s, a] += alpha * (target - q[s, a])
            s, a = s2, a2
            ret, hits, steps = ret + r, hits + hit, steps + 1
        log.append({"episode": ep, "return": ret, "collisions": hits, "steps": steps, "reached_goal": done,
                    "epsilon": eps})
    return q, pd.DataFrame(log)


def run_tabular_experiment(cfg: dict, smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Run every bump_prob x method x schedule x gamma x seed. Returns (curves, runs, value-iteration table, paths).

    curves: one row per training episode; runs: one row per trained agent with its greedy route, the exact expected
    return of its greedy policy from S (policy evaluation) and V*(S); paths: (bump_prob, method, schedule, gamma,
    seed) -> tiles of the greedy path.
    """
    seeds, episodes = list(cfg["seeds"]), cfg["episodes"]
    if smoke:
        seeds, episodes = seeds[: cfg["smoke"]["seeds"]], cfg["smoke"]["episodes"]
    curves, runs, vi_rows, paths = [], [], [], {}
    for bump in cfg["bump_probs"]:
        env = AisleGridworld(**{**cfg.get("grid", {}), "bump_prob": bump})
        optimal = {}
        for gamma in cfg["gammas"]:
            v, pi = value_iteration(env, gamma)
            optimal[gamma] = v[env.start]
            vi_rows.append({"bump_prob": bump, "gamma": gamma, "v_star_start": v[env.start],
                            "optimal_route": env.route(pi), "optimal_path_len": len(env.greedy_path(pi)) - 1})
        for method in METHODS:
            for sched_name, schedule in cfg["schedules"].items():
                for gamma in cfg["gammas"]:
                    for seed in seeds:
                        t0 = time.perf_counter()
                        q, log = train(env, method, episodes, cfg["alpha"], gamma, schedule,
                                       np.random.default_rng(seed))
                        secs = time.perf_counter() - t0
                        pi = q.argmax(axis=1)
                        key = dict(bump_prob=bump, method=method, schedule=sched_name, gamma=gamma, seed=seed)
                        curves.append(log.assign(**key))
                        paths[tuple(key.values())] = env.greedy_path(pi)
                        tail = log.tail(min(100, episodes))
                        runs.append({**key, "route": env.route(pi), "path_len": len(env.greedy_path(pi)) - 1,
                                     "greedy_value": policy_evaluation(env, pi, gamma)[env.start],
                                     "v_star_start": optimal[gamma],
                                     "train_return_last100": tail["return"].mean(),
                                     "collisions_last100": tail["collisions"].mean(),
                                     "collisions_total": int(log["collisions"].sum()),
                                     "env_steps": int(log["steps"].sum()), "train_seconds": secs})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(runs), pd.DataFrame(vi_rows), paths


def summarise_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over seeds per bump_prob x method x schedule x gamma, plus the share of seeds whose greedy route is the aisle."""
    g = runs.groupby(["bump_prob", "method", "schedule", "gamma"])
    cols = ["train_return_last100", "collisions_last100", "collisions_total", "greedy_value", "path_len",
            "env_steps", "train_seconds"]
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out["aisle_share"] = g["route"].apply(lambda r: (r == "aisle").mean())
    out["detour_share"] = g["route"].apply(lambda r: (r == "detour").mean())
    out["v_star_start"] = g["v_star_start"].first()
    out["seeds"] = g.size()
    return out.reset_index()


def modal_paths(paths: dict, bump_prob: float, schedule: str, gamma: float) -> dict[str, list[tuple[int, int]]]:
    """The most common greedy path across seeds, per method, for one cell of the experiment."""
    out = {}
    for method in METHODS:
        cands = [tuple(p) for (b, m, s, g, _), p in paths.items()
                 if (b, m, s, g) == (bump_prob, method, schedule, gamma)]
        out[method] = list(max(set(cands), key=cands.count))
    return out
