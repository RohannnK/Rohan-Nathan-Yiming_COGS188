"""Generic best-first search: BFS, UCS, greedy best-first and A* (uninformed and informed search).

One search core over (x, y, facing) with actions FORWARD, LEFT, RIGHT at unit cost; the algorithms differ only in
the priority they give a node: BFS uses FIFO order with an early goal test, UCS uses g, greedy uses h and A* uses
g + h (ties broken towards deeper nodes).
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .grid_graph import DELTA, FORWARD, LEFT, RIGHT, TURN
from .heuristics import manhattan_turn


@dataclass
class SearchResult:
    found: bool
    path: list[tuple] = field(default_factory=list)  # states from start to goal
    actions: list[int] = field(default_factory=list)  # len(path) - 1 rware actions
    cost: int = 0
    nodes_expanded: int = 0
    ms: float = 0.0
    fallback: bool = False  # space-time search fell back to spatial A*


def successors(s, mask: np.ndarray, blocked):
    """(action, next_state) pairs from state (x, y, facing); FORWARD only onto passable, unblocked cells."""
    x, y, d = s[:3]
    dx, dy = DELTA[d]
    nx_, ny = x + dx, y + dy
    h, w = mask.shape
    if 0 <= nx_ < w and 0 <= ny < h and mask[ny, nx_] and (nx_, ny) not in blocked:
        yield FORWARD, (nx_, ny, d)
    yield LEFT, (x, y, TURN[d, LEFT])
    yield RIGHT, (x, y, TURN[d, RIGHT])


def reconstruct(parent, s):
    path, actions = [s], []
    while parent[s] is not None:
        s, a = parent[s]
        path.append(s)
        actions.append(a)
    return path[::-1], actions[::-1]


def search(start, goal, mask, algorithm: str = "astar", blocked=frozenset(), heuristic=manhattan_turn) -> SearchResult:
    """Plan from start (x, y, facing) to any facing at goal (x, y). algorithm: bfs, ucs, greedy or astar."""
    t0 = perf_counter()
    start = tuple(start[:3])
    goal = tuple(goal)
    parent = {start: None}
    nodes = 0

    def done(s):
        path, actions = reconstruct(parent, s)
        return SearchResult(True, path, actions, len(actions), nodes, (perf_counter() - t0) * 1e3)

    if start[:2] == goal:
        return done(start)

    if algorithm == "bfs":
        frontier = deque([start])
        while frontier:
            s = frontier.popleft()
            nodes += 1
            for a, s2 in successors(s, mask, blocked):
                if s2 not in parent:
                    parent[s2] = (s, a)
                    if s2[:2] == goal:  # early goal test is exact for BFS with unit costs
                        return done(s2)
                    frontier.append(s2)
        return SearchResult(False, nodes_expanded=nodes, ms=(perf_counter() - t0) * 1e3)

    def priority(g, s):
        if algorithm == "ucs":
            return g
        if algorithm == "greedy":
            return heuristic(s, goal)
        if algorithm == "astar":
            return g + heuristic(s, goal)
        raise ValueError(algorithm)

    tie = itertools.count()
    frontier = [(priority(0, start), 0, next(tie), start)]
    best_g = {start: 0}
    closed = set()
    while frontier:
        _, neg_g, _, s = heapq.heappop(frontier)
        if s in closed:
            continue
        closed.add(s)
        nodes += 1
        if s[:2] == goal:
            return done(s)
        g = -neg_g + 1
        for a, s2 in successors(s, mask, blocked):
            if s2 not in closed and g < best_g.get(s2, 1 << 30):
                best_g[s2] = g
                parent[s2] = (s, a)
                heapq.heappush(frontier, (priority(g, s2), -g, next(tie), s2))
    return SearchResult(False, nodes_expanded=nodes, ms=(perf_counter() - t0) * 1e3)
