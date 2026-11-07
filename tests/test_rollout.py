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
