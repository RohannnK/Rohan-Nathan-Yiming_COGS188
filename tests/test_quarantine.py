import numpy as np
import pandas as pd

from monitor_helpers import CHOKE, GOALS, HIGHWAYS, episode
from warehouse_sabotage.agents.controller import build_fleet
from warehouse_sabotage.monitor.features import FEATURES, window_table
from warehouse_sabotage.monitor.quarantine import QuarantineMonitor
from warehouse_sabotage.types import RobotState, WarehouseState

SLOTS = [(x, y) for y in range(HIGHWAYS.shape[0]) for x in range(HIGHWAYS.shape[1]) if not HIGHWAYS[y, x]]


def state(t, robots, requested=(0,), carrying=None):
    """tiny_choke snapshot: a shelf on every slot, robots as (x, y, carrying_shelf_id or None)."""
    shelves = dict(enumerate(SLOTS))
    for i, (x, y, s) in enumerate(robots):
        if s is not None:
            shelves[s] = (x, y)
    return WarehouseState(
        t=t, robots=tuple(RobotState(i, x, y, 1, s) for i, (x, y, s) in enumerate(robots)),
        shelf_positions=shelves, requested_shelf_ids=tuple(requested), highways=HIGHWAYS,
        goals=tuple(GOALS), chokepoints=CHOKE)


class Flagging:
    """Stand-in monitor that has already flagged the given robots."""
    def __init__(self, ids):
        self.ids = frozenset(ids)
    def reset(self, state): pass
    def update(self, state, actions): pass
    def scores(self): return {}
    def flagged(self): return self.ids


def first_route(oracle, monitor=None, saboteur=(3, 5), quarantine=None, start=(3, 2)):
    """Honest robot 0 carries a requested shelf from ``start`` to a goal; robot 1 is the (parked) saboteur."""
    fleet = build_fleet("parked", 4, 1, {}, {}, oracle=oracle, monitor=monitor)
    if quarantine:
        fleet.ctx.quarantine.update(quarantine)
    s = state(0, [(*start, 0), (*saboteur, None), (0, 9, None), (9, 9, None)])
    fleet.reset(s, [np.random.default_rng(i) for i in range(4)])
    fleet.act(s)
    return fleet, [tuple(c) for c in fleet.policies[0].debug_info()["planned_cells"]]


def test_oracle_default_routes_around_the_saboteur_from_step_zero():
    fleet, route = first_route(oracle=True)
    assert route and (3, 5) not in route and not fleet.ctx.quarantine


def test_blind_fleet_drives_into_the_saboteur():
    _, route = first_route(oracle=False)
    assert (3, 5) in route


def test_detection_changes_the_honest_route():
    _, blind = first_route(oracle=False)
    fleet, flagged = first_route(oracle=False, monitor=Flagging({1}))
    assert (3, 5) in blind and (3, 5) not in flagged and flagged != blind
    assert fleet.ctx.quarantine == {1: (3, 5)}
    assert flagged[-1] != (3, 2)  # still has a route to follow


def test_quarantined_tile_stays_blocked_after_the_saboteur_leaves():
    _, route = first_route(oracle=False, saboteur=(6, 5), quarantine={1: (3, 5)})
    assert route and (3, 5) not in route and (6, 5) not in route[:3]


def test_monitor_flags_are_permanent_and_need_persistence():
    class Model:
        tau = 0.5
        def __init__(self): self.calls = 0
        def score(self, w):
            self.calls += 1
            return np.where(w["robot_id"] == 1, float(self.calls != 2), 0.0)  # robot 1 high, low, high

    paths = [[(0, 0)] * 60, [(3, 5)] * 60, [(6, 0)] * 60, [(9, 0)] * 60]
    for persist, expected in ((1, {1}), (2, set())):
        m = QuarantineMonitor(Model(), persist=persist)
        s0 = state(0, [(*p[0], None) for p in paths])
        m.reset(s0)
        for t in range(41):  # windows end at 19, 29, 39: scores high, low, high
            m.update(state(t, [(*p[t], None) for p in paths]), [0] * 4)
        assert m.flagged() == expected


def test_online_features_match_the_logged_features():
    n = 45
    p3 = [(0, 2)] * 6 + [(0, y) for y in range(3, 10)] + [(0, 9)] * (n - 13)
    paths = [[(0, 0), (3, 0)] * (n // 2 + 1), [(3, 5)] * n, [(6, y) for y in range(1, 10)] * 5, p3]
    paths = [p[:n] for p in paths]
    rng = np.random.default_rng(3)
    actions = {r: list(rng.integers(0, 5, n)) for r in range(4)}
    planned = {2: [[[6, 9], [5, 9]] if t % 3 == 0 else [] for t in range(n)]}
    target = {2: (5, 10), 3: (0, 9)}

    def carried(t):
        return 7 if 6 <= t <= 13 else None

    log = episode(paths, actions=actions, planned=planned)
    log["carrying_shelf_id"] = pd.array([carried(t) if r == 3 else None for r, t in zip(log["robot_id"], log["t"])],
                                        dtype="Int16")
    log.loc[(log["robot_id"] == 3) & (log["t"] == 5), "picked_up"] = True
    log.loc[(log["robot_id"] == 3) & (log["t"] == 13), "dropped"] = True
    for k, c in zip(("target_x", "target_y"), (0, 1)):
        log[k] = [target[r][c] if r in target else np.nan for r in log["robot_id"]]

    class Recorder:
        tau, seen = 2.0, []
        def score(self, w):
            self.seen.append(w)
            return np.zeros(len(w))

    rec = Recorder()
    now = [0]
    m = QuarantineMonitor(rec, debug=lambda: [{"planned_cells": planned.get(r, [[]] * n)[now[0]],
                                               "target": target.get(r)} for r in range(4)])
    m.reset(state(0, [(*p[0], carried(0) if r == 3 else None) for r, p in enumerate(paths)]))
    for t in range(41):
        now[0] = t
        m.update(state(t, [(*p[t], carried(t) if r == 3 else None) for r, p in enumerate(paths)]),
                 [actions[r][t] for r in range(4)])
    offline = window_table(log, HIGHWAYS, GOALS, ends=[19, 29, 39])
    online = pd.concat(rec.seen, ignore_index=True)
    assert list(online["t_end"].unique()) == [19, 29, 39]
    key = ["robot_id", "t_end"]
    pd.testing.assert_frame_equal(online.sort_values(key)[key + list(FEATURES)].reset_index(drop=True),
                                  offline.sort_values(key)[key + list(FEATURES)].reset_index(drop=True))
    assert online[online["robot_id"] == 3]["drop_displacement"].max() == 7


def test_quarantined_goal_is_not_chosen_as_target():
    _, route = first_route(oracle=False, saboteur=(6, 5), quarantine={1: (4, 10)}, start=(4, 8))
    assert (4, 10) not in route and (5, 10) in route


def test_fleet_never_enters_quarantined_tiles_even_if_every_goal_is_quarantined():
    _, route = first_route(oracle=False, saboteur=(6, 5), quarantine={1: (4, 10), 2: (5, 10)}, start=(4, 8))
    assert (4, 10) not in route and (5, 10) not in route


def test_quarantined_tile_can_be_left_by_a_robot_standing_on_it():
    _, route = first_route(oracle=False, saboteur=(6, 5), quarantine={1: (3, 2)}, start=(3, 2))
    assert route and (3, 2) not in route[1:]
