"""Week 8 tabular experiment: python scripts/run_tabular.py --config configs/week8_tabular.yaml [--smoke]"""
import argparse
from pathlib import Path

import yaml

from warehouse_sabotage.rl.gridworld import AisleGridworld
from warehouse_sabotage.rl.tabular import modal_paths, run_tabular_experiment, summarise_runs
from warehouse_sabotage.viz.plots import plot_learning_curves, plot_paths

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/week8_tabular.yaml")
p.add_argument("--smoke", action="store_true", help="2 seeds, 50 episodes, no figures")
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
curves, runs, vi, paths = run_tabular_experiment(cfg, smoke=args.smoke)
summary = summarise_runs(runs)
out = Path("results") / (f"{cfg['name']}_smoke" if args.smoke else "summary")
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / f"{cfg['name']}.csv", index=False)
vi.to_csv(out / f"{cfg['name']}_value_iteration.csv", index=False)
print(vi.round(2).to_string(index=False))
cols = ["bump_prob", "method", "schedule", "gamma", "aisle_share", "train_return_last100_mean",
        "collisions_last100_mean", "deadlock_rate_last100_mean", "collisions_total_mean", "greedy_value_mean", "v_star_start"]
print(summary[cols].round(3).to_string(index=False))
if not args.smoke:
    media, g = Path("docs/media"), cfg["report_gamma"]
    plot_learning_curves(curves, "return", "return per episode (20-episode mean)", media / "week8_returns.png", g,
                         ylim={s: (-60, -5) for s in cfg["schedules"]})
    plot_learning_curves(curves, "collisions", "collisions per episode (20-episode mean)",
                         media / "week8_collisions.png", g, ylim={"constant": (0, 1)})
    for bump in cfg["bump_probs"]:
        plot_paths(AisleGridworld(**cfg["grid"]).layout, modal_paths(paths, bump, "constant", g),
                   media / f"week8_paths_bump{bump}.png",
                   f"Most common greedy path over seeds (constant epsilon, gamma {g}, bump p = {bump})")
print(f"wrote {out}")
