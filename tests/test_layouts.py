import networkx as nx
import pytest

from warehouse_sabotage.env.layouts import LAYOUTS, chokepoints, parse_layout
from warehouse_sabotage.env.make_env import make_env


def highway_graph(highways):
    h, w = highways.shape
    g = nx.Graph()
    tiles = [(x, y) for y in range(h) for x in range(w) if highways[y, x]]
    g.add_nodes_from(tiles)
    for x, y in tiles:
        for nx_, ny in ((x + 1, y), (x, y + 1)):
            if nx_ < w and ny < h and highways[ny, nx_]:
                g.add_edge((x, y), (nx_, ny))
    return g


@pytest.mark.parametrize("name", ["tiny", "tiny_choke"])
def test_layout_loads_in_rware(name):
    env = make_env({"layout": name})
    env.reset(seed=0)
    w = env.unwrapped
    highways, goals = parse_layout(LAYOUTS[name])
    assert w.grid_size == (11, 10)
    assert (w.highways == highways).all()
    assert tuple(w.goals) == goals == ((4, 10), (5, 10))


@pytest.mark.parametrize("name", ["tiny", "tiny_choke", "small"])
def test_removing_any_chokepoint_keeps_highways_connected(name):
    highways, goals = parse_layout(LAYOUTS[name])
    g = highway_graph(highways)
    assert nx.is_connected(g)
    for c in chokepoints(highways, goals):
        g2 = g.copy()
        g2.remove_node(c)
        assert nx.is_connected(g2), c


def test_chokepoints_include_goal_access_tiles():
    highways, goals = parse_layout(LAYOUTS["tiny_choke"])
    cps = chokepoints(highways, goals)
    assert {(4, 10), (5, 10), (4, 9), (5, 9)} <= cps
    assert {(3, 5), (6, 5), (0, 5), (9, 5)} <= cps  # the four vertical corridors
    stock = chokepoints(*parse_layout(LAYOUTS["tiny"]))
    assert {(4, 9), (5, 9), (3, 10), (6, 10)} <= stock
    assert (4, 5) not in stock  # open middle band


def test_small_layout_matches_rware_stock_small():
    import gymnasium as gym

    stock = gym.make("rware-small-4ag-v2", disable_env_checker=True).unwrapped
    stock.reset(seed=0)
    env = make_env({"layout": "small"})
    env.reset(seed=0)
    w = env.unwrapped
    highways, goals = parse_layout(LAYOUTS["small"])
    assert w.grid_size == stock.grid_size == (20, 10)
    assert (w.highways == stock.highways).all() and (highways == stock.highways).all()
    assert tuple(w.goals) == goals == tuple(tuple(g) for g in stock.goals) == ((4, 19), (5, 19))
    cps = chokepoints(highways, goals)
    assert {(3, 4), (6, 4), (4, 18), (5, 18)} <= cps  # top-block corridors and goal access tiles
    assert (4, 13) not in cps  # open bottom band
