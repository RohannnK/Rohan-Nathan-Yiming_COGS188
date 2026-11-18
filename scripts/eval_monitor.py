"""Closed-loop Week 7 evaluation: python scripts/eval_monitor.py [--config configs/week7_eval.yaml] [--smoke]

Runs every condition of the config (monitors from data/models/) and writes results/summary/<name>.csv (fleet metrics,
throughput retained as ratio_to_no_saboteur) and <name>_monitor.csv (detection rate, TTD, robot-episode FPR).
"""
import argparse
import logging
from pathlib import Path

import yaml

from warehouse_sabotage.eval.experiment import run_experiment
from warehouse_sabotage.eval.metrics import condition_summary, monitor_summary

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/week7_eval.yaml")
p.add_argument("--smoke", action="store_true", help="2 episodes per condition, 100 steps")
args = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg = yaml.safe_load(open(args.config))
episodes = run_experiment(cfg, smoke=args.smoke)
fleet, detection = condition_summary(episodes), monitor_summary(episodes, cfg["env"]["max_steps"])
out = Path("results") / f"{cfg['name']}_smoke" if args.smoke else Path("results/summary")
out.mkdir(parents=True, exist_ok=True)
fleet.to_csv(out / f"{cfg['name']}.csv", index=False)
detection.to_csv(out / f"{cfg['name']}_monitor.csv", index=False)
cols = ["condition", "layout", "episodes", "deliveries_mean", "deliveries_std", "ratio_to_no_saboteur", "deadlock_rate"]
print(fleet[cols].round(2).to_string(index=False))
print(detection.round(2).to_string(index=False))
