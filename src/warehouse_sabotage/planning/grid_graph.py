"""Passability for loaded and unloaded robots (state-space formulation for search).

States for search are (x, y, facing) with rware Direction values UP=0 (y-1), DOWN=1 (y+1), LEFT=2 (x-1), RIGHT=3 (x+1).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..types import WarehouseState

NOOP, FORWARD, LEFT, RIGHT, TOGGLE_LOAD = range(5)
DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
_CW = (0, 3, 1, 2)  # UP, RIGHT, DOWN, LEFT: rware's RIGHT turns clockwise through this list
TURN = {(d, a): _CW[(_CW.index(d) + (1 if a == RIGHT else -1)) % 4] for d in range(4) for a in (LEFT, RIGHT)}
UNREACHABLE = 10**6


def standing_shelves(state: WarehouseState) -> set[tuple[int, int]]:
    """Tiles holding a shelf that no robot is carrying."""
    carried = {r.carrying_shelf_id for r in state.robots}
    return {p for s, p in state.shelf_positions.items() if s not in carried}


def passable(state: WarehouseState, loaded: bool) -> np.ndarray:
    """[y, x] bool mask ignoring robots. Unloaded robots drive anywhere; loaded robots avoid standing shelves."""
    mask = np.ones(state.highways.shape, dtype=bool)
    if loaded:
        for x, y in standing_shelves(state):
            mask[y, x] = False
    return mask


def bfs_distances(mask: np.ndarray, sources) -> np.ndarray:
    """Multi-source 4-neighbour BFS distance ignoring facing; UNREACHABLE where not reachable."""
    h, w = mask.shape
    dist = np.full((h, w), UNREACHABLE, dtype=np.int64)
    q = deque()
    for x, y in sources:
        dist[y, x] = 0
        q.append((x, y))
    while q:
        x, y = q.popleft()
        for dx, dy in DELTA.values():
            nx_, ny = x + dx, y + dy
            if 0 <= nx_ < w and 0 <= ny < h and mask[ny, nx_] and dist[ny, nx_] == UNREACHABLE:
                dist[ny, nx_] = dist[y, x] + 1
                q.append((nx_, ny))
    return dist
