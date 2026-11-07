"""Reservation table and space-time A* with WAIT (cooperative pathfinding, Silver 2005).

Honest robots plan one at a time against each other's reservations (prioritised planning). A move a -> b during
t -> t+1 conflicts if another robot holds the vertex (t+1, b) or the reverse edge b -> a over the same step (rware
makes head-on swaps fail). Beyond the horizon the planner falls back to spatial A* without reservations.
"""
from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from time import perf_counter

from .grid_graph import DELTA, FORWARD, LEFT, NOOP, RIGHT, TURN
from .heuristics import manhattan_turn
from .search import SearchResult, reconstruct, search


class ReservationTable:
    def __init__(self):
        self.vertex: dict[tuple, int] = {}  # (t, cell) -> robot
        self.edge: dict[tuple, int] = {}  # (t, from_cell, to_cell) -> robot
        self.keys: dict[int, list] = defaultdict(list)

    def _add(self, table, key, rid):
        if table.setdefault(key, rid) == rid:
            self.keys[rid].append((table, key))

    def reserve_path(self, rid: int, t0: int, cells: list, stay: int = 0) -> None:
        """Reserve cells[i] at time t0 + i, the edges between them, and the last cell for `stay` more steps."""
        for i, c in enumerate(cells):
            self._add(self.vertex, (t0 + i, c), rid)
            if i and c != cells[i - 1]:
                self._add(self.edge, (t0 + i - 1, cells[i - 1], c), rid)
        for k in range(1, stay + 1):
            self._add(self.vertex, (t0 + len(cells) - 1 + k, cells[-1]), rid)

    def clear(self, rid: int) -> None:
        for table, key in self.keys.pop(rid, []):
            if table.get(key) == rid:
                del table[key]

    def move_ok(self, rid: int, t: int, a, b) -> bool:
        """Can robot rid go from cell a (time t) to cell b (time t+1)? a == b is a WAIT or turn."""
        owner = self.vertex.get((t + 1, b), rid)
        if owner != rid:
            return False
        return a == b or self.edge.get((t, b, a), rid) == rid

    def owner(self, t: int, cell):
        return self.vertex.get((t, cell))


def space_time_astar(start, t0: int, goal, mask, blocked, table: ReservationTable, rid: int,
                     horizon: int = 32) -> SearchResult:
    """A* over (x, y, facing, t) with FORWARD/LEFT/RIGHT/WAIT; falls back to spatial A* if the goal lies beyond
    the horizon or cannot be reached conflict-free within it."""
    t_start = perf_counter()
    goal = tuple(goal)
    h, w = mask.shape
    s0 = (*start[:3], t0)
    parent = {s0: None}
    tie = itertools.count()
    frontier = [(manhattan_turn(s0, goal), 0, next(tie), s0)]
    closed = set()
    nodes = 0
    while frontier:
        _, neg_g, _, s = heapq.heappop(frontier)
        if s in closed:
            continue
        closed.add(s)
        nodes += 1
        x, y, d, t = s
        if (x, y) == goal:
            path, actions = reconstruct(parent, s)
            return SearchResult(True, [p[:3] for p in path], actions, len(actions), nodes,
                                (perf_counter() - t_start) * 1e3)
        if t - t0 >= horizon:
            continue
        g = -neg_g + 1
        moves = []
        dx, dy = DELTA[d]
        nx_, ny = x + dx, y + dy
        if 0 <= nx_ < w and 0 <= ny < h and mask[ny, nx_] and (nx_, ny) not in blocked:
            moves.append((FORWARD, (nx_, ny, d, t + 1)))
        if table.move_ok(rid, t, (x, y), (x, y)):
            moves += [(LEFT, (x, y, TURN[d, LEFT], t + 1)), (RIGHT, (x, y, TURN[d, RIGHT], t + 1)),
                      (NOOP, (x, y, d, t + 1))]
        for a, s2 in moves:
            if s2 in closed or s2 in parent:
                continue
            if a == FORWARD and not table.move_ok(rid, t, (x, y), (nx_, ny)):
                continue
            parent[s2] = (s, a)
            heapq.heappush(frontier, (g + manhattan_turn(s2, goal), -g, next(tie), s2))
    r = search(start, goal, mask, "astar", blocked)
    r.nodes_expanded += nodes
    r.ms = (perf_counter() - t_start) * 1e3
    r.fallback = True
    return r
