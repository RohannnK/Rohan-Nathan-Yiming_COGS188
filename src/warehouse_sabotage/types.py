"""Shared interfaces between teammates (agents and environments): state snapshot, Policy and Monitor protocols.

Fields may grow but never silently shrink; change only through a PR all three people see (the shared contract between monitor, RL and planning code).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotState:
    robot_id: int  # 0-indexed, matches env.unwrapped.agents order
    x: int
    y: int
    direction: int  # rware Direction value 0..3 (UP, DOWN, LEFT, RIGHT)
    carrying_shelf_id: int | None  # 0-indexed shelf id or None


@dataclass(frozen=True)
class WarehouseState:
    t: int
    robots: tuple[RobotState, ...]
    shelf_positions: dict[int, tuple[int, int]]  # shelf_id -> (x, y)
    requested_shelf_ids: tuple[int, ...]
    highways: np.ndarray  # [y, x] uint8, treat as read-only
    goals: tuple[tuple[int, int], ...]
    chokepoints: frozenset[tuple[int, int]]


class Policy(Protocol):
    def reset(self, state: WarehouseState, robot_id: int, rng: np.random.Generator) -> None: ...
    def act(self, state: WarehouseState) -> int: ...  # rware action 0..4
    def debug_info(self) -> dict: ...  # phase, plan_len, plan_ms, nodes_expanded, replanned, sabotage_active


class Monitor(Protocol):
    def reset(self, state: WarehouseState) -> None: ...
    def update(self, state: WarehouseState, actions: Sequence[int]) -> None: ...
    def scores(self) -> dict[int, float]: ...  # robot_id -> P(saboteur)
    def flagged(self) -> frozenset[int]: ...
