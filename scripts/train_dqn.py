"""Week 8 DQN honest robot.

Train:  python scripts/train_dqn.py --config configs/week8_dqn.yaml [--seed 8100] [--smoke]
Report: python scripts/train_dqn.py --config configs/week8_dqn.yaml --report   (after all seeds have trained)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from warehouse_sabotage.rl.dqn import dqn_report, train_dqn
from warehouse_sabotage.viz.training import plot_dqn_progress

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/week8_dqn.yaml")
p.add_argument("--seed", type=int, help="train one seed (default: every seed in the config)")
p.add_argument("--smoke", action="store_true", help="1500 steps, 1 evaluation episode")
p.add_argument("--report", action="store_true", help="summarise trained seeds and draw the figure")
args = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg = yaml.safe_load(open(args.config))
run_dir = Path("results") / (f"{cfg['name']}_smoke" if args.smoke else cfg["name"])
if args.report:
    curves = pd.concat([pd.read_csv(f) for f in sorted(run_dir.glob("curves_seed_*.csv"))], ignore_index=True)
    summary, astar, noop = dqn_report(cfg, curves)
    table_dir, fig_dir = (run_dir, run_dir) if args.smoke else (Path("results/summary"), Path("docs/media"))
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_dir / f"{cfg['name']}.csv", index=False)
    plot_dqn_progress(curves, astar, noop, fig_dir / f"{cfg['name']}_progress.png")
    print(summary.round(3).to_string(index=False))
else:
    for seed in [args.seed] if args.seed is not None else cfg["seeds"]:
        curves = train_dqn(cfg, seed, run_dir, smoke=args.smoke)
        curves.to_csv(run_dir / f"curves_seed_{seed}.csv", index=False)
        print(f"seed {seed}: {len(curves)} episodes, {curves['step'].iloc[-1]} steps, "
              f"{curves['seconds'].iloc[-1]:.0f} s")
