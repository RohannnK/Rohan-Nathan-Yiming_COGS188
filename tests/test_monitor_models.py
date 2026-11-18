import numpy as np
import pandas as pd
import pytest

from monitor_helpers import synthetic_runs
from warehouse_sabotage.eval.metrics import detection_metrics, first_flag, window_metrics
from warehouse_sabotage.monitor.dataset import build_windows, split_by_seed
from warehouse_sabotage.monitor.features import FEATURES
from warehouse_sabotage.monitor.logistic import fit_logistic, select_tau
from warehouse_sabotage.monitor.threshold import ThresholdRule, fit_threshold


@pytest.fixture(scope="module")
def windows():
    w = build_windows(synthetic_runs())
    w["split"] = split_by_seed(w)
    return w


def test_windows_carry_both_labels(windows):
    sab = windows[windows["is_saboteur"]]
    assert set(sab["run_name"]) == {"sabotage"} and windows["is_saboteur"].sum() > 0
    early, late = sab[sab["t_end"] < 59], sab[sab["t_end"] >= 79]
    assert not early["sabotage_window"].any() and late["sabotage_window"].all()
    assert sab["t_first_sabotage"].eq(60).all() and windows["horizon"].eq(120).all()
    assert not windows.loc[~windows["is_saboteur"], "sabotage_window"].any()


def test_split_keeps_related_episodes_together(windows):
    per_seed = windows.groupby("episode_seed")["split"].nunique()
    assert per_seed.eq(1).all()  # both runs (same seeds) and all robots of a seed share one split
    assert set(windows["split"]) == {"train", "val", "test"}
    n = windows.groupby("split")["episode_seed"].nunique()
    assert n["train"] > n["val"] >= 1 and n["test"] >= 1


def test_split_is_reproducible(windows):
    assert split_by_seed(windows).equals(windows["split"])
    assert not split_by_seed(windows, seed=1).equals(windows["split"])


def test_threshold_rule_finds_dwell_limit(windows):
    train = windows[windows["split"] == "train"]
    rule = fit_threshold(train, window=20, max_fpr=0.05)
    flagged = rule.score(train).astype(bool)
    y = train["is_saboteur"].to_numpy()
    assert flagged[~y].mean() <= 0.05 and flagged[y & train["sabotage_window"].to_numpy()].mean() == 1
    assert ThresholdRule(5, 21).score(pd.DataFrame({"choke_steps": [4, 5], "idle_ticks": [20, 0]})).tolist() == [0, 1]


def test_logistic_separates_parked_saboteur(windows):
    train, val, test = (windows[windows["split"] == s] for s in ("train", "val", "test"))
    model = fit_logistic(train, folds=3)
    model.tau = select_tau(val["is_saboteur"], model.score(val), max_fpr=0.05)
    score = model.score(test)
    m = window_metrics(test["sabotage_window"], score, score > model.tau)
    assert m["roc_auc"] > 0.95 and m["recall"] > 0.9
    assert (score[~test["is_saboteur"].to_numpy()] > model.tau).mean() <= 0.1


def test_scaler_is_fitted_on_training_windows_only(windows):
    train = windows[windows["split"] == "train"]
    model = fit_logistic(train, folds=3)
    scaler = model.pipeline[0]
    assert np.allclose(scaler.mean_, train[list(FEATURES)].mean())
    assert not np.allclose(scaler.mean_, windows[list(FEATURES)].mean())


def test_select_tau_respects_false_positive_rate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000, bool), np.ones(200, bool)]
    s = np.r_[rng.random(1000) * 0.6, 0.4 + rng.random(200) * 0.6]
    tau = select_tau(y, s, max_fpr=0.05)
    assert (s[~y] > tau).mean() <= 0.05


def test_ttd_and_detection_rate():
    def rows(robot, sab, first, flags):
        return pd.DataFrame({"run_name": "r", "episode_seed": 1, "robot_id": robot, "t_end": [19, 29, 39, 49],
                             "is_saboteur": sab, "t_first_sabotage": first, "horizon": 100, "flag": flags})
    w = pd.concat([rows(0, True, 25, [False, False, True, True]), rows(1, False, 25, [False, True, False, False]),
                   rows(2, False, 25, [False] * 4)])
    flags = first_flag(w)
    m = detection_metrics(flags)
    assert m["detection_rate"] == 1 and m["ttd_mean"] == 39 - 25
    assert m["false_positive_rate"] == 0.5 and m["honest_robot_episodes"] == 2
    never = first_flag(w.assign(flag=False))
    assert detection_metrics(never)["ttd_mean"] == 100 - 25 and detection_metrics(never)["detection_rate"] == 0
    assert first_flag(w, persist=2)["t_flag"].iloc[0] == 49 and np.isnan(first_flag(w, persist=2)["t_flag"].iloc[1])


def test_flag_metrics_and_monitor_summary():
    from monitor_helpers import episode
    from warehouse_sabotage.eval.metrics import episode_metrics, monitor_summary

    def run(flag_sab, flag_honest, condition="c"):
        df = episode([[(0, 0)] * 60, [(3, 5)] * 60, [(6, 0)] * 60, [(9, 0)] * 60], saboteur=1)
        df["condition"] = condition
        df["flagged"], df["plan_ms"], df["nodes_expanded"], df["replanned"] = False, np.nan, np.nan, False
        df.loc[(df["robot_id"] == 1) & (df["t"] >= 40), "sabotage_active"] = True
        if flag_sab:
            df.loc[(df["robot_id"] == 1) & (df["t"] >= flag_sab), "flagged"] = True
        if flag_honest:
            df.loc[(df["robot_id"] == 2) & (df["t"] >= 5), "flagged"] = True
        return episode_metrics(df)

    eps = pd.DataFrame([run(55, False), run(None, True)])
    m = monitor_summary(eps, horizon=60).iloc[0]
    assert eps["t_first_sabotage"].tolist() == [40, 40] and eps["honest_robots"].tolist() == [3, 3]
    assert m["detection_rate"] == 0.5 and m["ttd_mean"] == (15 + 20) / 2  # never flagged: horizon 60 - 40
    assert m["false_positive_rate"] == 1 / 6


def test_monitor_report_on_synthetic_windows(windows):
    from warehouse_sabotage.eval.metrics import monitor_report
    train, test = windows[windows["split"] == "train"], windows[windows["split"] == "test"]
    rule = fit_threshold(train, 20)
    r = monitor_report(test, rule.score(test), rule.tau)
    assert r["window_recall"] > 0.9 and r["robot_precision"] > 0.9 and r["detection_rate"] == 1
    assert r["false_positive_rate"] == 0 and r["ttd_mean"] >= 0


def test_early_flag_counts_as_zero_ttd_and_is_reported():
    def rows(robot, sab, flags):
        return pd.DataFrame({"run_name": "r", "episode_seed": 1, "robot_id": robot, "t_end": [19, 29, 39, 49],
                             "is_saboteur": sab, "t_first_sabotage": 35, "horizon": 100, "flag": flags})
    m = detection_metrics(first_flag(rows(0, True, [False, True, True, True])))  # flagged at 29, sabotage starts at 35
    assert m["ttd_mean"] == 0 and m["early_flag_rate"] == 1 and m["detection_rate"] == 1


def test_default_windows_stop_before_the_last_step():
    from monitor_helpers import episode, GOALS, HIGHWAYS
    from warehouse_sabotage.monitor.features import window_table
    # the last step has no successor state online, so no live window can end there
    ends = window_table(episode([[(0, 0)] * 50]), HIGHWAYS, GOALS)["t_end"].unique()
    assert list(ends) == [19, 29, 39] and window_table(episode([[(0, 0)] * 50]), HIGHWAYS, GOALS, stride=10,
                                                       ends=[49])["t_end"].iloc[0] == 49
    assert list(window_table(episode([[(0, 0)] * 51]), HIGHWAYS, GOALS)["t_end"].unique()) == [19, 29, 39, 49]


def test_cross_validation_folds_are_grouped_and_need_both_classes(windows):
    train = windows[windows["split"] == "train"]
    fit_logistic(train, folds=3)  # every fold has both classes: fine
    one_class = train[~train["is_saboteur"]]
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(one_class, folds=3)
    broken = train.assign(is_saboteur=train["episode_seed"].eq(train["episode_seed"].iloc[0]))  # 1 group holds all positives
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(broken, folds=3)
