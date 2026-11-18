"""Fit and evaluate the threshold and logistic monitors on logged episodes (Week 7).

python scripts/train_monitor.py [--prefix week7_data] [--window 20] [--stride 10] [--persist 1]
python scripts/train_monitor.py --selection episode --suffix _v2     # Version 2
Reads data/episodes/<prefix>__*/, writes results/summary/week7_monitor<suffix>.csv, week7_coefficients<suffix>.csv and
the two monitors to data/models/. Version 1 (--selection window) fixes the window false-positive rate; Version 2
(--selection episode) chooses the threshold and persistence on the validation split by robot-episode false-positive rate.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from warehouse_sabotage.eval.logging import load_run
from warehouse_sabotage.eval.metrics import monitor_report
from warehouse_sabotage.monitor.dataset import build_windows, split_by_seed
from warehouse_sabotage.monitor.logistic import fit_logistic, select_tau
from warehouse_sabotage.monitor.quarantine import save_monitor
from warehouse_sabotage.monitor.selection import diagnose, select_flagging
from warehouse_sabotage.monitor.threshold import ThresholdRule, fit_threshold

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="week7_data")
p.add_argument("--data", default="data")
p.add_argument("--window", type=int, default=20)
p.add_argument("--stride", type=int, default=10)
p.add_argument("--persist", type=int, default=1, help="consecutive flagged windows needed to flag a robot")
p.add_argument("--max-fpr", type=float, default=0.05,
               help="false-positive rate allowed when choosing limits (per window, or per robot-episode with --selection episode)")
p.add_argument("--selection", choices=["window", "episode"], default="window",
               help="window: Version 1; episode: replay windows in time order and choose tau and persist (Version 2)")
p.add_argument("--persists", type=int, nargs="+", default=[1, 2, 3, 4, 6], help="persistence values tried by --selection episode")
p.add_argument("--min-detection", type=float, default=0.5, help="detection rate below which a selection counts as not useful")
p.add_argument("--suffix", default="", help="appended to every output name, so Version 1 files are kept")
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

model = fit_logistic(train, seed=args.seed)
if args.selection == "window":
    rule = fit_threshold(val, args.window, args.max_fpr)
    model.tau = select_tau(val["is_saboteur"], model.score(val), args.max_fpr)
    persist = {"threshold": args.persist, "logistic": args.persist}
    trials = pd.DataFrame()
else:
    val_score = model.score(val)
    taus = np.unique(np.append(np.quantile(val_score, np.linspace(0.5, 0.9995, 100)), val_score.max()))  # last one never flags
    grid = [(c, i) for c in range(args.window + 2) for i in range(args.window + 2)]  # limit window + 1 never fires
    chosen, trials = {}, []
    for name, cands in (("logistic", [(t, val_score > t) for t in taus]),
                        ("threshold", [(g, ThresholdRule(*g).score(val) > 0.5) for g in grid])):
        setting, n, table, useful = select_flagging(val, cands, args.max_fpr, args.persists, args.min_detection)
        chosen[name] = (setting, n)
        trials.append(table.assign(monitor=name, setting=table["setting"].astype(str)))
        log.info("%s selection on validation: %s", name, diagnose(table, args.max_fpr))
        if not useful:
            log.warning("%s: no setting with robot-episode FPR <= %.2f detects at least %.0f%% of saboteurs on "
                        "validation (diagnostic, not a project requirement); report this as a failure", name,
                        args.max_fpr, 100 * args.min_detection)
    model.tau = float(chosen["logistic"][0])
    rule = ThresholdRule(*chosen["threshold"][0])
    persist = {name: chosen[name][1] for name in chosen}
    trials = pd.concat(trials, ignore_index=True)
log.info("threshold rule: %s, persist %d; logistic tau = %.3f, persist %d", rule, persist["threshold"], model.tau,
         persist["logistic"])

rows = [{"monitor": name, "split": split, "persist": persist[name],
         **monitor_report(w, m.score(w), m.tau, persist[name])}
        for name, m in (("threshold", rule), ("logistic", model)) for split, w in (("val", val), ("test", test))]
out = Path("results/summary")
out.mkdir(parents=True, exist_ok=True)
report = pd.DataFrame(rows)
report.to_csv(out / f"week7_monitor{args.suffix}.csv", index=False)
model.coefficients().rename("coefficient").to_csv(out / f"week7_coefficients{args.suffix}.csv")
if len(trials):
    trials.to_csv(out / f"week7_selection{args.suffix}.csv", index=False)  # every (setting, persist) tried on validation
for name, m in (("threshold", rule), ("logistic", model)):
    save_monitor(Path(args.data) / "models" / f"{name}{args.suffix}.joblib", m, args.window, args.stride, persist[name])
print(report.round(3).T.to_string())
print(model.coefficients().sort_values().round(2).to_string())
