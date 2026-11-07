import itertools

import numpy as np
import pytest

from warehouse_sabotage.planning.cooperative import ReservationTable, space_time_astar
from warehouse_sabotage.planning.grid_graph import FORWARD, NOOP
from warehouse_sabotage.planning.heuristics import manhattan_turn
from warehouse_sabotage.planning.search import search, successors

GRIDS = {
    "open": """
.....
.....
.....
""",
    "walls": """
.#...
.#.#.
...#.
##.#.
.....
""",
    "corridors": """
.......
.##.##.
.##.##.
.......
""",
    "maze": """
..#....
#.#.##.
..#..#.
.##..#.
....#..
""",
}


def mask_of(text):
    return np.array([[c == "." for c in line] for line in text.strip().split("\n")])


def cases(mask, n, seed=0):
    rng = np.random.default_rng(seed)
    cells = [(x, y) for y, x in zip(*np.nonzero(mask))]
    for _ in range(n):
        s, g = rng.choice(len(cells), 2, replace=False)
        yield (*cells[s], int(rng.integers(4))), cells[g]


@pytest.mark.parametrize("name", GRIDS)
def test_astar_matches_ucs_cost_and_expands_no_more(name):
    mask = mask_of(GRIDS[name])
    for start, goal in cases(mask, 40):
        ucs = search(start, goal, mask, "ucs")
        astar = search(start, goal, mask, "astar")
        bfs = search(start, goal, mask, "bfs")
        greedy = search(start, goal, mask, "greedy")
        assert ucs.found == astar.found == bfs.found == greedy.found
        if not ucs.found:
            continue
        assert astar.cost == ucs.cost == bfs.cost
        assert greedy.cost >= ucs.cost
        assert astar.nodes_expanded <= ucs.nodes_expanded
        assert astar.path[-1][:2] == goal and len(astar.actions) == astar.cost


@pytest.mark.parametrize("name", GRIDS)
def test_heuristic_is_consistent_and_admissible(name):
    mask = mask_of(GRIDS[name])
    h, w = mask.shape
    states = [(x, y, d) for y in range(h) for x in range(w) for d in range(4) if mask[y, x]]
    for goal in [(x, y) for y in range(h) for x in range(w) if mask[y, x]][::3]:
        for s in states:
            if s[:2] == goal:
                assert manhattan_turn(s, goal) == 0
            for _, s2 in successors(s, mask, frozenset()):
                assert manhattan_turn(s, goal) <= 1 + manhattan_turn(s2, goal)
        for s in states[::7]:
            r = search(s, goal, mask, "ucs")
            if r.found:
                assert manhattan_turn(s, goal) <= r.cost


def test_blocked_cells_are_avoided():
    mask = mask_of(GRIDS["open"])
    r = search((0, 1, 3), (4, 1), mask, "astar", blocked=frozenset({(2, 1)}))
    assert r.found and (2, 1) not in [p[:2] for p in r.path]


def simulate(plans):
    """Positions of each plan over time (a finished robot stays at its last cell)."""
    T = max(len(p) for p in plans)
    return [[p[min(t, len(p) - 1)][:2] for p in plans] for t in range(T)]


def assert_conflict_free(plans):
    traj = simulate(plans)
    for t, cells in enumerate(traj):
        assert len(set(cells)) == len(cells), f"vertex conflict at t={t}: {cells}"
        if t:
            for i, j in itertools.combinations(range(len(cells)), 2):
                assert not (traj[t - 1][i] == cells[j] and traj[t - 1][j] == cells[i]), f"swap at t={t}"


def test_space_time_astar_avoids_vertex_and_swap_conflicts():
    # 1-wide corridor with a side pocket at (4, 1): robots heading in opposite directions must use it.
    mask = mask_of("""
.......
####.##
""")
    table = ReservationTable()
    a = space_time_astar((0, 0, 3), 0, (6, 0), mask, frozenset(), table, 0, horizon=32)
    table.reserve_path(0, 0, [p[:2] for p in a.path], stay=4)
    b = space_time_astar((6, 0, 2), 0, (0, 0), mask, frozenset(), table, 1, horizon=32)
    assert a.found and b.found and not b.fallback
    assert (4, 1) in [p[:2] for p in b.path]
    assert_conflict_free([a.path, b.path])


def test_reservation_table_edge_swap():
    table = ReservationTable()
    table.reserve_path(0, 5, [(1, 0), (0, 0)])
    assert not table.move_ok(1, 5, (0, 0), (1, 0))  # head-on swap
    assert not table.move_ok(1, 5, (2, 0), (0, 0))  # vertex (6, (0, 0)) taken
    assert table.move_ok(0, 5, (1, 0), (0, 0))  # own reservations never conflict
    table.clear(0)
    assert table.move_ok(1, 5, (0, 0), (1, 0))


def test_space_time_falls_back_to_spatial_beyond_horizon():
    mask = mask_of(GRIDS["corridors"])
    r = space_time_astar((0, 0, 3), 0, (6, 3), mask, frozenset(), ReservationTable(), 0, horizon=3)
    assert r.found and r.path[-1][:2] == (6, 3)
    assert r.actions.count(FORWARD) == 9
