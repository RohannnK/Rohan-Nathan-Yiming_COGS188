"""Manhattan and Manhattan + turn heuristics (admissible and consistent heuristics).

States are (x, y, facing). Every action costs 1, so Manhattan distance is a lower bound on FORWARD moves, and at
least one turn is needed when the goal is not straight ahead. Consistency is checked exhaustively in the tests.
"""
from __future__ import annotations


def manhattan(s, goal) -> int:
    return abs(goal[0] - s[0]) + abs(goal[1] - s[1])


def manhattan_turn(s, goal) -> int:
    """Manhattan distance plus 1 if the goal cannot be reached without turning."""
    dx, dy = goal[0] - s[0], goal[1] - s[1]
    m = abs(dx) + abs(dy)
    if m == 0:
        return 0
    if dx and dy:
        return m + 1
    facing_needed = 3 if dx > 0 else 2 if dx < 0 else 1 if dy > 0 else 0  # RIGHT, LEFT, DOWN, UP
    return m + (s[2] != facing_needed)
