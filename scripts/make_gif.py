"""Render one episode per condition to docs/media/<condition>_<layout>.gif (needs a display; macOS works).

python scripts/make_gif.py --conditions no_saboteur goal_blocker --layout tiny_choke
"""
import argparse

import yaml

from warehouse_sabotage.eval.experiment import episode_specs
from warehouse_sabotage.viz.gif import record_gif

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/week6_baseline.yaml")
p.add_argument("--conditions", nargs="+", default=["no_saboteur", "goal_blocker"])
p.add_argument("--layout", default="tiny_choke")
p.add_argument("--seed", type=int, default=None, help="default: first seed in the config")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--every", type=int, default=2, help="keep every n-th frame")
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
cfg["env"] = {**cfg.get("env", {}), "max_steps": args.steps}
for spec, seeds in episode_specs(cfg):
    if spec["condition"] in args.conditions and spec["layout"] == args.layout:
        path = record_gif(spec, args.seed or seeds[0], f"docs/media/{spec['condition']}_{spec['layout']}.gif",
                          every=args.every, max_frames=args.steps // args.every)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
