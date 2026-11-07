"""Greedy task assignment with a shared claim set (multi-agent coordination).

Idle honest robots claim the nearest unclaimed requested shelf; goal, slot and parking targets are also claimed so
two honest robots never aim for the same tile.
"""
from __future__ import annotations

from ..planning.grid_graph import UNREACHABLE, bfs_distances, passable, standing_shelves
from ..planning.heuristics import manhattan
from ..types import WarehouseState


class TaskBoard:
    def __init__(self):
        self.shelf: dict[int, int] = {}  # robot -> claimed requested shelf id
        self.tile: dict[int, tuple[int, int]] = {}  # robot -> claimed goal / slot / parking tile

    def release(self, rid: int) -> None:
        self.shelf.pop(rid, None)
        self.tile.pop(rid, None)

    def claimed_shelves(self, exclude: int | None = None) -> set[int]:
        return {s for r, s in self.shelf.items() if r != exclude}

    def taken_tiles(self, exclude: int) -> set[tuple[int, int]]:
        return {t for r, t in self.tile.items() if r != exclude}

    def claim_shelf(self, rid: int, state: WarehouseState, avoid_tiles=frozenset()) -> int | None:
        """Greedy: nearest (Manhattan) requested shelf that nobody claimed, carries, or sits on via `avoid_tiles`."""
        me = state.robots[rid]
        carried = {r.carrying_shelf_id for r in state.robots}
        taken = self.claimed_shelves(exclude=rid) | carried
        options = [s for s in state.requested_shelf_ids
                   if s not in taken and state.shelf_positions[s] not in avoid_tiles]
        if not options:
            self.shelf.pop(rid, None)
            return None
        best = min(options, key=lambda s: (manhattan((me.x, me.y), state.shelf_positions[s]), s))
        self.shelf[rid] = best
        return best

    def claim_tile(self, rid: int, state: WarehouseState, candidates, loaded: bool, key=None):
        """Nearest reachable candidate tile (BFS) not claimed by another robot nor occupied by one."""
        me = state.robots[rid]
        occupied = {(r.x, r.y) for r in state.robots if r.robot_id != rid}
        free = [c for c in candidates if c not in occupied and c not in self.taken_tiles(rid)]
        if not free:
            free = list(candidates)
        if not free:
            return None
        mask = passable(state, loaded)
        for x, y in free:
            mask[y, x] = True
        dist = bfs_distances(mask, [(me.x, me.y)])
        best = min(free, key=key or (lambda c: (dist[c[1], c[0]], c)))
        if dist[best[1], best[0]] >= UNREACHABLE:
            return None
        self.tile[rid] = best
        return best


def empty_slots(state: WarehouseState) -> list[tuple[int, int]]:
    """Non-highway tiles without a standing shelf: where a loaded robot can drop its shelf."""
    shelves = standing_shelves(state)
    h, w = state.highways.shape
    return [(x, y) for y in range(h) for x in range(w) if not state.highways[y, x] and (x, y) not in shelves]


def parking_tiles(state: WarehouseState) -> list[tuple[int, int]]:
    """Non-highway tiles under a non-requested shelf: out of every loaded robot's way."""
    requested = {state.shelf_positions[s] for s in state.requested_shelf_ids}
    shelves = standing_shelves(state)
    return [p for p in shelves if p not in requested]
