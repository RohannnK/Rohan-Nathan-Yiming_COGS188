"""Run the Week 6 baseline: python scripts/run_baseline.py --config configs/week6_baseline.yaml [--smoke]"""
import argparse
import logging
from pathlib import Path

import yaml

from warehouse_sabotage.eval.experiment import run_experiment
from warehouse_sabotage.eval.metrics import condition_summary

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/week6_baseline.yaml")
p.add_argument("--smoke", action="store_true", help="2 episodes per condition, 100 steps")
args = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg = yaml.safe_load(open(args.config))
summary = condition_summary(run_experiment(cfg, smoke=args.smoke))
out = Path("results") / f"{cfg['name']}_smoke" if args.smoke else Path("results/summary")
out.mkdir(parents=True, exist_ok=True)
path = out / ("summary.csv" if args.smoke else f"{cfg['name']}.csv")
summary.to_csv(path, index=False)
cols = ["condition", "layout", "episodes", "deliveries_mean", "deliveries_std", "ratio_to_no_saboteur", "deadlock_rate"]
print(summary[cols].round(2).to_string(index=False))
print(f"wrote {path}")
