import pytest

from warehouse_sabotage.eval.logging import rows_to_frame
from warehouse_sabotage.eval.metrics import episode_metrics, longest_gap
from warehouse_sabotage.eval.rollout import run_episode


def episode(kind, layout="tiny_choke", steps=150, seed=5):
    spec = {"run_name": "test", "condition": kind, "layout": layout, "kind": kind, "env": {"max_steps": steps}}
    return rows_to_frame(run_episode(spec, seed))


@pytest.mark.parametrize("kind", ["honest", "parked", "goal_blocker", "corridor_parker", "shelf_hider", "shelf_squatter"])
def test_logged_episode_is_consistent(kind):
    df = episode(kind)
    assert len(df) == 150 * 4 and df.groupby("t").size().eq(4).all()
    assert not df.duplicated(["t", "x", "y"]).any()  # never two robots on one tile
    nxt = df.sort_values(["robot_id", "t"]).groupby("robot_id").shift(-1)
    cur = df.sort_values(["robot_id", "t"])
    has_next = nxt["t"].notna()
    moved = (cur["x"] != nxt["x"]) | (cur["y"] != nxt["y"])
    assert (cur.loc[has_next, "moved"] == moved[has_next]).all()
    picked = cur["picked_up"] & has_next
    assert cur.loc[picked, "carrying_shelf_id"].isna().all() and nxt.loc[picked, "carrying_shelf_id"].notna().all()
    delivered = cur["delivered"] & has_next
    assert set(zip(nxt.loc[delivered, "x"], nxt.loc[delivered, "y"])) <= {(4, 10), (5, 10)}
    assert not df.loc[~df["is_saboteur"], "sabotage_active"].any()
    n_sab = df.groupby("t")["is_saboteur"].sum()
    assert (n_sab == (0 if kind in ("honest", "parked") else 1)).all()


def test_honest_fleet_delivers_and_random_does_not():
    assert episode_metrics(episode("honest"))["deliveries"] >= 10
    assert episode_metrics(episode("random"))["deliveries"] <= 2


def test_longest_gap():
    assert longest_gap([], 500) == 500
    assert longest_gap([9, 10, 400], 500) == 389


def test_planned_cells_start_with_next_tile():
    df = episode("goal_blocker")
    assert df["planned_cells"].map(len).max() <= 8
    others = df[df["is_saboteur"]]
    assert others["planned_cells"].map(len).eq(0).all()
    nxt = df.sort_values(["robot_id", "t"]).groupby("robot_id").shift(-1)
    cur = df.sort_values(["robot_id", "t"])
    following = cur["planned_cells"].map(len).gt(0) & cur["moved"] & nxt["t"].notna()
    assert following.sum() > 50
    first = cur.loc[following, "planned_cells"].map(lambda p: tuple(p[0]))
    assert (first == list(zip(nxt.loc[following, "x"], nxt.loc[following, "y"]))).all()


def test_week6_log_schema_has_week7_columns():
    df = episode("goal_blocker", steps=60)
    assert {"target_x", "target_y", "flagged"} <= set(df.columns) and not df["flagged"].any()
    assert df.loc[df["is_saboteur"], "target_x"].notna().any()


def test_monitor_flags_reach_the_log_and_the_planners(tmp_path):
    from warehouse_sabotage.monitor.quarantine import save_monitor
    from warehouse_sabotage.monitor.threshold import ThresholdRule

    path = save_monitor(tmp_path / "rule.joblib", ThresholdRule(choke_limit=5, idle_limit=5), window=20, stride=10)
    spec = {"run_name": "test", "condition": "c", "layout": "tiny_choke", "kind": "corridor_parker", "oracle": False,
            "monitor": str(path), "env": {"max_steps": 200}}
    df = rows_to_frame(run_episode(spec, 5))
    flagged = df[df["flagged"]]
    assert len(flagged) == 0 or flagged["t"].min() >= 20  # nothing can be flagged before the first full window
    first = flagged.groupby("robot_id")["t"].min()
    assert all(df[(df["robot_id"] == r) & (df["t"] >= t)]["flagged"].all() for r, t in first.items())  # permanent
