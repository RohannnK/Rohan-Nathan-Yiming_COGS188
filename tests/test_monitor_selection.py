import numpy as np
import pandas as pd

from monitor_helpers import synthetic_runs
from warehouse_sabotage.eval.metrics import detection_metrics, first_flag
from warehouse_sabotage.monitor.dataset import build_windows
from warehouse_sabotage.monitor.selection import Replay, diagnose, select_flagging


def robot(rid, flags, sab=False, first=40, seed=1):
    return pd.DataFrame({"run_name": "r", "episode_seed": seed, "robot_id": rid, "t_end": 19 + 10 * np.arange(len(flags)),
                         "is_saboteur": sab, "t_first_sabotage": first if sab else np.nan, "horizon": 100,
                         "flag": flags})


def test_persistence_needs_consecutive_windows_in_time_order():
    w = robot(0, [True, False, True, True, False])
    for persist, expected in ((1, 19), (2, 49), (3, np.nan)):
        t = Replay(w).flags(w["flag"], persist)["t_flag"].iloc[0]
        assert (np.isnan(t) and np.isnan(expected)) or t == expected
    shuffled = w.sample(frac=1, random_state=0)  # row order must not matter, only t_end
    assert Replay(shuffled).flags(shuffled["flag"], 2)["t_flag"].iloc[0] == 49


def test_replay_matches_first_flag_and_resets_between_robots():
    w = build_windows(synthetic_runs(n_seeds=6))
    flag = np.random.default_rng(0).random(len(w)) < 0.4
    replay = Replay(w)
    for persist in (1, 2, 3):
        got = replay.flags(flag, persist).sort_values(["run_name", "episode_seed", "robot_id"], ignore_index=True)
        want = first_flag(w.assign(flag=flag), persist=persist).sort_values(
            ["run_name", "episode_seed", "robot_id"], ignore_index=True)
        assert np.array_equal(got["t_flag"], want["t_flag"], equal_nan=True)


def test_episode_false_positive_rate_counts_robots_not_windows():
    w = pd.concat([robot(0, [False, True, False, False], sab=True), robot(1, [False, True, False, False]),
                   robot(2, [False] * 4), robot(3, [True] * 4)])
    m = detection_metrics(Replay(w).flags(w["flag"], 1))
    assert m["honest_robot_episodes"] == 3 and m["false_positive_rate"] == 2 / 3  # window FPR would be 5/12
    assert detection_metrics(Replay(w).flags(w["flag"], 2))["false_positive_rate"] == 1 / 3  # only robot 3 persists


def scores(honest_spike=0.6):
    """Three honest robots (one with a single high window) and a saboteur that scores high for six windows."""
    w = pd.concat([robot(0, [0] * 8, sab=True), robot(1, [0] * 8), robot(2, [0] * 8), robot(3, [0] * 8)], ignore_index=True)
    s = np.full(len(w), 0.1)
    s[(w["robot_id"] == 0).to_numpy() & (w["t_end"] >= 39).to_numpy()] = 0.9
    s[(w["robot_id"] == 1).to_numpy() & (w["t_end"] == 59).to_numpy()] = honest_spike
    return w, s


def test_selection_keeps_fpr_low_and_does_not_choose_a_silent_monitor():
    w, s = scores()
    cands = [(tau, s > tau) for tau in (0.05, 0.5, 0.95)]
    tau, persist, table, useful = select_flagging(w, cands, max_fpr=0.05, persists=(1, 2))
    assert useful and tau != 0.95  # tau 0.95 never flags: zero false positives, zero detection
    chosen = table[(table["setting"] == tau) & (table["persist"] == persist)].iloc[0]
    assert chosen["false_positive_rate"] <= 0.05 and chosen["detection_rate"] == 1
    assert table[(table["setting"] == 0.5) & (table["persist"] == 1)]["false_positive_rate"].iloc[0] > 0.05


def test_selection_reports_when_nothing_useful_meets_the_constraint():
    w, s = scores(honest_spike=0.9)  # honest robot 1 looks exactly like the saboteur for six windows
    w.loc[(w["robot_id"] == 1), "flag"] = 0
    s[(w["robot_id"] == 1).to_numpy() & (w["t_end"] >= 39).to_numpy()] = 0.9
    s[(w["robot_id"] == 2).to_numpy() & (w["t_end"] >= 39).to_numpy()] = 0.9
    s[(w["robot_id"] == 3).to_numpy() & (w["t_end"] >= 39).to_numpy()] = 0.9
    _, _, table, useful = select_flagging(w, [(t, s > t) for t in (0.05, 0.5, 0.95)], max_fpr=0.05, persists=(1, 2))
    assert not useful and table.loc[table["false_positive_rate"] <= 0.05, "detection_rate"].max() == 0


def test_empty_feasible_set_is_reported_not_crashed():
    # real validation scores: every candidate tau sits below honest robots' high windows, no never-flagging tau offered
    w, s = scores()
    s[(w["robot_id"] > 0).to_numpy()] = 0.9  # all three honest robots score as high as the saboteur
    tau, persist, table, useful = select_flagging(w, [(t, s > t) for t in (0.05, 0.5, 0.8)], max_fpr=0.05, persists=(1, 2))
    assert not useful and table["false_positive_rate"].min() > 0.05
    chosen = table[(table["setting"] == tau) & (table["persist"] == persist)].iloc[0]
    assert chosen["false_positive_rate"] == table["false_positive_rate"].min()  # closest to the constraint, not relaxed
    d = diagnose(table, 0.05)
    assert d["candidates"] == 6 and d["feasible"] == 0 and np.isnan(d["best_detection_within_fpr"])
    assert d["min_fpr"] == table["false_positive_rate"].min()


def test_diagnose_with_a_feasible_set():
    w, s = scores()
    _, _, table, _ = select_flagging(w, [(t, s > t) for t in (0.05, 0.5, 0.95)], max_fpr=0.05, persists=(1, 2))
    d = diagnose(table, 0.05)
    assert d["feasible"] > 0 and d["best_detection_within_fpr"] == 1 and d["balanced_detection"] == 1
