from rware.warehouse import Direction

from warehouse_sabotage.env.make_env import make_env
from warehouse_sabotage.env.reward_wrapper import SabotageRewardWrapper
from warehouse_sabotage.env.state import extract_state


def test_state_matches_grid_indexing():
    env = make_env({"layout": "tiny_choke"})
    env.reset(seed=3)
    w = env.unwrapped
    s = extract_state(w, 0)
    for r in s.robots:
        assert w.grid[0, r.y, r.x] == r.robot_id + 1  # grid ids are 1-indexed, [y, x]
    for sid, (x, y) in s.shelf_positions.items():
        assert w.grid[1, y, x] == sid + 1
    assert len(s.requested_shelf_ids) == 4
    assert all(not s.highways[y, x] for x, y in s.shelf_positions.values())
    assert (4, 9) in s.chokepoints


def _place_for_delivery(w, robot):
    """Put `robot` on (4, 9) facing DOWN, carrying the first requested shelf."""
    shelf = w.request_queue[0]
    for a in w.agents:  # clear the tiles we need
        if (a.x, a.y) in {(4, 9), (4, 10)} and a is not w.agents[robot]:
            a.x, a.y = 0, a.id  # distinct free tiles in column x=0
    a = w.agents[robot]
    a.x, a.y, a.dir, a.carrying_shelf = 4, 9, Direction.DOWN, shelf
    shelf.x, shelf.y = 4, 9
    w._recalc_grid()


def test_honest_delivery_rewards_and_events():
    env = SabotageRewardWrapper(make_env({"layout": "tiny"}), has_saboteur=True)
    _, info = env.reset(seed=11)
    k = info["saboteur_id"]
    honest = (k + 1) % 4
    w = env.unwrapped
    _place_for_delivery(w, honest)
    actions = [0] * 4
    actions[honest] = 1  # FORWARD onto goal (4, 10)
    _, rewards, _, _, info = env.step(actions)
    ev = info["events"]
    assert ev["delivered"][honest] == 1 and ev["moved"][honest]
    assert rewards[honest] == 10 - 0.01
    assert rewards[k] == -10


def test_saboteur_id_is_seeded_and_varies():
    env = SabotageRewardWrapper(make_env({"layout": "tiny"}), has_saboteur=True)
    ids = [env.reset(seed=s)[1]["saboteur_id"] for s in range(20)]
    assert ids == [env.reset(seed=s)[1]["saboteur_id"] for s in range(20)]
    assert len(set(ids)) > 1
    assert SabotageRewardWrapper(make_env({}), has_saboteur=False).reset(seed=0)[1]["saboteur_id"] is None
