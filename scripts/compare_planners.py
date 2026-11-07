"""BFS vs UCS vs greedy vs A* on random start/goal pairs: python scripts/compare_planners.py [--pairs 100]"""
import argparse
from pathlib import Path

from warehouse_sabotage.eval.experiment import compare_planners, summarise_planners

p = argparse.ArgumentParser()
p.add_argument("--pairs", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

summary = summarise_planners(compare_planners(args.pairs, args.seed))
path = Path("results/summary/planner_comparison.csv")
path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(path, index=False)
print(summary.round(3).to_string(index=False))
print(f"wrote {path}")
