"""Paired delivery differences with bootstrap 95% intervals, from the logged closed-loop episodes (no simulation).

python scripts/paired_analysis.py [--config configs/week7_eval_v2.yaml] [--data data]
The config may list its own comparisons as ``pairs: [[treatment, control], ...]``.
Reads data/episodes/<name>__<condition>__<layout>/, recomputes the per-episode metrics with eval.metrics.episode_metrics
and writes results/summary/<name>_paired.csv. It also prints how many steps every condition ran, so episodes that
ended before max_steps would show up.
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from warehouse_sabotage.eval.logging import load_run
from warehouse_sabotage.eval.metrics import episode_metrics, paired_difference

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/week7_eval_v2.yaml")
p.add_argument("--data", default="data")
p.add_argument("--resamples", type=int, default=10000)
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
name, rows = cfg["name"], []
for layout in cfg["layouts"]:
    for cond in cfg["conditions"]:
        ep = load_run(args.data, f"{name}__{cond}__{layout}")
        rows += [episode_metrics(g, cfg["deadlock_L"][layout]) for _, g in ep.groupby("episode_seed")]
episodes = pd.DataFrame(rows)

steps = episodes.groupby(["condition", "layout"], sort=False)["steps"].agg(["min", "mean", "max"])
short = steps[steps["min"] < cfg["env"]["max_steps"]]
print(f"conditions with an episode shorter than max_steps ({cfg['env']['max_steps']}): {len(short)} of {len(steps)}")
print(short.to_string() if len(short) else "none")

if "pairs" in cfg:  # [[treatment, control], ...]
    pairs = [tuple(p) for p in cfg["pairs"]]
else:  # Week 7 Version 2 comparisons
    pairs = [(f"{b}_{k}", f"{b}_blind") for b in ("goal_blocker", "corridor_parker", "shelf_hider", "shelf_squatter")
             for k in ("logistic", "threshold", "oracle")]
    pairs += [("no_saboteur_logistic", "no_saboteur"), ("no_saboteur_threshold", "no_saboteur"),
              ("idle_robot_logistic", "idle_robot_blind"), ("idle_robot_threshold", "idle_robot_blind")]
out = pd.concat([paired_difference(episodes, t, c, resamples=args.resamples) for t, c in pairs], ignore_index=True)
Path("results/summary").mkdir(parents=True, exist_ok=True)
out.to_csv(f"results/summary/{name}_paired.csv", index=False)
print(out.round(2).to_string(index=False))
