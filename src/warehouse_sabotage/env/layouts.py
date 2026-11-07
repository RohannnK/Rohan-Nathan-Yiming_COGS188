"""Layout strings and chokepoint detection (environment design, partial observability).

Layouts use rware's string format: ``x`` shelf slot, ``.`` highway, ``g`` goal (also a highway).
Positions are ``(x, y)``; arrays are indexed ``[y, x]``.
"""
from __future__ import annotations

import numpy as np

# Identical to rware's stock tiny map (shelf_columns=3, shelf_rows=1, column_height=8).
TINY = """
..........
.xx....xx.
.xx....xx.
.xx....xx.
.xx....xx.
.xx....xx.
.xx....xx.
.xx....xx.
.xx....xx.
..........
....gg....
"""

# Middle band filled with shelves: four 1-wide vertical corridors (x=0, 3, 6, 9).
TINY_CHOKE = """
..........
.xx.xx.xx.
.xx.xx.xx.
.xx.xx.xx.
.xx.xx.xx.
.xx.xx.xx.
.xx.xx.xx.
.xx.xx.xx.
.xx.xx.xx.
..........
xxxxggxxxx
"""

LAYOUTS = {"tiny": TINY, "tiny_choke": TINY_CHOKE}


def parse_layout(layout: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Return (highways[y, x] uint8, goals) exactly as rware parses a layout string."""
    lines = layout.strip().replace(" ", "").split("\n")
    highways = np.array([[c in ".g" for c in line] for line in lines], dtype=np.uint8)
    goals = tuple((x, y) for y, line in enumerate(lines) for x, c in enumerate(line) if c == "g")
    return highways, goals


def chokepoints(highways: np.ndarray, goals) -> frozenset[tuple[int, int]]:
    """Operational chokepoints: highway tiles in a 1-wide corridor for loaded robots, goals and goal access tiles.

    A highway tile is in a 1-wide corridor if both its left and right neighbours, or both its up and down
    neighbours, are non-highway (shelf slot or map edge). Loaded robots can only use highways at layout time.
    """
    h, w = highways.shape

    def hw(x, y):
        return 0 <= x < w and 0 <= y < h and bool(highways[y, x])

    out = {
        (x, y)
        for y in range(h)
        for x in range(w)
        if hw(x, y) and ((not hw(x - 1, y) and not hw(x + 1, y)) or (not hw(x, y - 1) and not hw(x, y + 1)))
    }
    for gx, gy in goals:
        out.add((gx, gy))
        out |= {(gx + dx, gy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if hw(gx + dx, gy + dy)}
    return frozenset(out)
