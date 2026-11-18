"""Fit and evaluate the threshold and logistic monitors on logged episodes (Week 7).

python scripts/train_monitor.py [--prefix week7_data] [--window 20] [--stride 10] [--persist 1]
Reads data/episodes/<prefix>__*/, writes results/summary/week7_monitor.csv, week7_coefficients.csv and the two monitors
to data/models/.
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from warehouse_sabotage.eval.logging import load_run
from warehouse_sabotage.eval.metrics import monitor_report
from warehouse_sabotage.monitor.dataset import build_windows, split_by_seed
from warehouse_sabotage.monitor.logistic import fit_logistic, select_tau
from warehouse_sabotage.monitor.quarantine import save_monitor
from warehouse_sabotage.monitor.threshold import fit_threshold

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="week7_data")
p.add_argument("--data", default="data")
p.add_argument("--window", type=int, default=20)
p.add_argument("--stride", type=int, default=10)
p.add_argument("--persist", type=int, default=1, help="consecutive flagged windows needed to flag a robot")
p.add_argument("--max-fpr", type=float, default=0.05, help="window false-positive rate allowed when choosing limits")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_monitor")

runs = sorted(d.name for d in (Path(args.data) / "episodes").glob(f"{args.prefix}__*"))
windows, missing = [], []
for r in runs:
    ep = load_run(args.data, r)
    windows.append(build_windows(ep, args.window, args.stride))
    missing.append(ep.groupby("is_saboteur")["target_x"].apply(lambda s: s.isna().mean()))
windows = pd.concat(windows, ignore_index=True)
log.info("share of rows without a planner target (path_efficiency is 0 there), by is_saboteur: %s",
         pd.concat(missing).groupby(level=0).mean().round(4).to_dict())
windows["split"] = split_by_seed(windows, seed=args.seed)
log.info("%d windows from %d runs; episodes per split: %s", len(windows), len(runs),
         windows.groupby("split")["episode_seed"].nunique().to_dict())
train, val, test = (windows[windows["split"] == s] for s in ("train", "val", "test"))

rule = fit_threshold(val, args.window, args.max_fpr)
model = fit_logistic(train, seed=args.seed)
model.tau = select_tau(val["is_saboteur"], model.score(val), args.max_fpr)
log.info("threshold rule: %s; logistic tau = %.3f", rule, model.tau)

rows = [{"monitor": name, "split": split, **monitor_report(w, m.score(w), m.tau, args.persist)}
        for name, m in (("threshold", rule), ("logistic", model)) for split, w in (("val", val), ("test", test))]
out = Path("results/summary")
out.mkdir(parents=True, exist_ok=True)
report = pd.DataFrame(rows)
report.to_csv(out / "week7_monitor.csv", index=False)
model.coefficients().rename("coefficient").to_csv(out / "week7_coefficients.csv")
for name, m in (("threshold", rule), ("logistic", model)):
    save_monitor(Path(args.data) / "models" / f"{name}.joblib", m, args.window, args.stride, args.persist)
print(report.round(3).T.to_string())
print(model.coefficients().sort_values().round(2).to_string())
