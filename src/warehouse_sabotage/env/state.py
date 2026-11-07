"""Extract a WarehouseState snapshot from env.unwrapped (fully observable fleet-manager view).

rware stores 1-indexed ids in ``grid`` but 0-indexed ``agents``/``shelfs`` lists; ``Shelf.id - 1`` is the list index.
"""
from __future__ import annotations

from ..types import RobotState, WarehouseState


def extract_state(w, t: int) -> WarehouseState:
    """Snapshot of ``env.unwrapped`` at timestep t (state before the next action)."""
    highways = w.highways.view()
    highways.flags.writeable = False
    return WarehouseState(
        t=t,
        robots=tuple(
            RobotState(i, a.x, a.y, a.dir.value, a.carrying_shelf.id - 1 if a.carrying_shelf else None)
            for i, a in enumerate(w.agents)
        ),
        shelf_positions={s.id - 1: (s.x, s.y) for s in w.shelfs},
        requested_shelf_ids=tuple(s.id - 1 for s in w.request_queue),
        highways=highways,
        goals=tuple(tuple(g) for g in w.goals),
        chokepoints=w.chokepoints,
    )
