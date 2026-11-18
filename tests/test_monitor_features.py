import numpy as np
import pytest

from monitor_helpers import CHOKE, GOALS, HIGHWAYS, episode
from warehouse_sabotage.monitor.features import FEATURES, corridor_tiles, window_table

W = 20


def feats(df, **kw):
    out = window_table(df, HIGHWAYS, GOALS, window=W, stride=10, **kw)
    return {r: g.set_index("t_end") for r, g in out.groupby("robot_id")}


def test_corridor_tiles_exclude_goal_area():
    corridor = corridor_tiles(HIGHWAYS, GOALS)
    assert (3, 5) in corridor and (3, 5) in CHOKE
    assert (4, 9) in CHOKE and (4, 9) not in corridor  # goal access tile
    assert (4, 10) in CHOKE and (4, 10) not in corridor


def test_windows_are_full_length_with_stride():
    df = episode([[(0, 0)] * 45] * 2)
    f = feats(df)
    assert list(f[0].index) == [19, 29, 39]
    assert set(FEATURES) <= set(f[0].columns)


def test_parked_robot_in_corridor():
    df = episode([[(3, 5)] * 40, [(0, 0), (3, 0)] * 20])
    a = feats(df)[0].loc[19]
    assert a["corridor_frac"] == 1 and a["choke_steps"] == W and a["idle_ticks"] == W
    assert a["action_entropy"] == 0 and a["deliveries"] == 0 and a["aisle_blocking"] == 0
    b = feats(df)[1].loc[19]
    assert b["corridor_frac"] == 0 and b["idle_ticks"] == 0


def test_action_entropy_of_two_equal_actions_is_one_bit():
    df = episode([[(0, 0)] * 40], actions={0: [1, 2] * 20})
    assert feats(df)[0].loc[19, "action_entropy"] == pytest.approx(1.0)


def test_aisle_blocking_uses_other_robots_planned_cells():
    # robot 1 plans to walk down x=3 through (3,5), where robot 0 sits; robot 1 is not blocking anyone
    path1 = [(3, y) for y in range(1, 9)] + [(3, 8)] * 32
    planned = {1: [[[3, 5]] if t < 10 else [] for t in range(40)]}
    df = episode([[(3, 5)] * 40, path1], planned=planned)
    f = feats(df, ends=[19, 39])
    assert f[0].loc[19, "aisle_blocking"] == 10 and f[1].loc[19, "aisle_blocking"] == 0
    assert f[0].loc[39, "aisle_blocking"] == 0


def test_next_to_stalled_needs_a_robot_that_stopped_on_a_highway():
    stalled = [(6, 4)] * 40
    df = episode([[(6, 5)] * 40, stalled, [(0, 0)] * 40])
    f = feats(df)
    # robot 1 counts as stalled from its third step (STALL_STEPS), robot 0 and 1 are neighbours throughout
    assert f[0].loc[19, "next_to_stalled"] == W - 2 and f[1].loc[19, "next_to_stalled"] == W - 2
    assert f[2].loc[19, "next_to_stalled"] == 0
    # a robot parked under a shelf (not a highway tile) is not a stalled robot
    parked = episode([[(2, 4)] * 40, [(3, 4)] * 40])
    assert not HIGHWAYS[4, 2]
    assert feats(parked)[1].loc[19, "next_to_stalled"] == 0


def test_goal_distance_and_corridor_fraction():
    df = episode([[(4, 9)] * 20 + [(3, 9)] * 20])
    a = feats(df, ends=[19, 39])[0]
    assert a.loc[19, "goal_distance"] == 1 and a.loc[39, "goal_distance"] == 2


def test_carrying_and_delivery_counts():
    df = episode([[(0, 0)] * 40])
    df["carrying_shelf_id"] = df["carrying_shelf_id"].astype("Int16")
    df.loc[:9, "carrying_shelf_id"] = 3            # carries an unrequested shelf for 10 steps
    df.loc[20:25, "carrying_shelf_id"] = 4
    df.loc[20:25, "carrying_requested"] = True
    df.loc[19, ["picked_up"]] = True               # picks up at t=19; the shelf is requested at t=20
    df.loc[24, "delivered"] = True
    f = feats(df)[0]
    assert f.loc[19, "unrequested_carried"] == 10
    assert f.loc[19, "requested_undelivered"] == 1  # picked up in the window, not yet delivered
    assert f.loc[29, "deliveries"] == 1 and f.loc[29, "requested_undelivered"] == 0


def test_drop_displacement_is_pickup_to_drop_distance():
    df = episode([[(0, 0)] * 40])
    df.loc[2, "picked_up"] = True
    df.loc[2, ["x", "y"]] = (1, 1)
    df.loc[12, "dropped"] = True
    df.loc[12, ["x", "y"]] = (4, 4)
    assert feats(df)[0].loc[19, "drop_displacement"] == 6


def test_path_efficiency_is_progress_per_step():
    path = [(3, y) for y in range(1, 9)] + [(3, 8)] * 12  # 7 steps of progress in the first 8 rows, then waits
    df = episode([path + [(3, 8)] * 20])
    df["target_x"], df["target_y"] = 3, 8
    eff = feats(df)[0].loc[19, "path_efficiency"]
    assert eff == pytest.approx(7 / 19)
    df["target_x"] = np.nan
    assert feats(df)[0].loc[19, "path_efficiency"] == 0.0


def test_features_ignore_label_columns():
    df = episode([[(3, 5)] * 40, [(0, 0)] * 40], saboteur=0)
    df2 = df.copy()
    df2["sabotage_active"], df2["is_saboteur"] = True, False
    a = window_table(df, HIGHWAYS, GOALS)[list(FEATURES)]
    b = window_table(df2, HIGHWAYS, GOALS)[list(FEATURES)]
    assert a.equals(b)
