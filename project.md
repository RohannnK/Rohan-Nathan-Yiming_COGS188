# Project Guide: Catching a Covert Saboteur in an Autonomous Warehouse Fleet

COGS 188 (UC San Diego) group project. Team: Rohan Kaman, Nathan Bustos Medina, Yiming Sun.
Repo: https://github.com/RohannnK/Rohan-Nathan-Yiming_COGS188 (fork of the course template).

This file is the single source of truth for how the project is built. Read it fully before writing code. When this file and the proposal notebook disagree on *what* we promised, the proposal wins; when they disagree on *how* to build it, this file wins. If you change an interface, a metric definition or the folder layout, update this file in the same commit.

---

## 1. What we are building (one paragraph)

Four robots work in the rware (Robotic Warehouse) simulator. One of them is a covert saboteur that tries to lower fleet throughput without being caught: it parks in 1-tile corridors or on goal access tiles just as honest robots approach, sits under requested shelves, or moves requested shelves far from the packing stations. The honest robots are controlled by A* with replanning (the benchmark). A runtime monitor computes behavioural features over sliding windows, flags the saboteur (threshold rule vs. logistic regression), and the flagged robot's tile becomes a permanent obstacle so the honest fleet reroutes. Alongside this we run a controlled tabular experiment (SARSA vs. Q-learning on a small single-aisle gridworld) and a Double/Dueling DQN honest robot, and, as an extension, a DQN saboteur trained against the frozen A* + monitor fleet.

Core principle: **the classical/scripted pipeline must always produce a complete demo.** Every learned component is a comparison or an extension, never a dependency of the main demo.

---

## 2. Course concepts: where each one lives

The project must visibly use COGS 188 material. Every module's docstring should name the concept it implements, and the final report will use this table.

| Course topic (lecture) | Where it appears | Status |
|---|---|---|
| Agents and environments (week 1) | Honest robots, saboteur, monitor as distinct agents; multi-agent mixed-motive setting | Committed |
| Uninformed search: BFS, uniform-cost (week 1-2) | `planning/search.py` baselines for the planner comparison (nodes expanded, time) | Committed (Week 6) |
| Informed search: greedy best-first, A*, admissible and consistent heuristics | `planning/search.py`; A* over (x, y, facing) with Manhattan + turn heuristic | Committed (Week 6) |
| Cooperative planning with reservation tables (Silver 2005) | `planning/cooperative.py`; space-time A* with a WAIT action | Committed (Week 6) |
| Local search: hill climbing vs. simulated annealing (L04) | Tuning monitor window length and threshold in Week 9 | Committed (Week 9, small) |
| Stochasticity and partial observability (L05) | Each robot sees a 3x3 window (sensor_range=1); random spawns and request queue; hidden saboteur identity | Committed |
| Adversarial games (L06) | Saboteur vs. fleet as a mixed-motive, imperfect-information game; learned saboteur trained against a fixed opponent | Committed framing, extension for learning |
| Bandits and exploration vs. exploitation (L08) | epsilon-greedy schedules for SARSA, Q-learning and DQN; compare at least two decay schedules | Committed (Weeks 8-9) |
| MDPs (L09-10) | Formal MDP for the single-aisle gridworld (states, actions, transitions with stochastic blocker, rewards, gamma) | Committed (Week 8) |
| Dynamic programming (L11) | Value iteration on the single-aisle gridworld as the exact optimal policy that SARSA and Q-learning are compared against | Recommended addition (cheap, Week 8) |
| Monte Carlo methods, on-policy vs. off-policy (L13-14) | Optional on-policy MC control on the same gridworld | Optional |
| TD learning, SARSA vs. Q-learning, discounting | `rl/tabular.py`; gamma sweep | Committed (Week 8) |
| Neural networks as function approximators | `rl/dqn.py`, Double + Dueling DQN | Committed (Weeks 8-9) |
| Logistic regression (outside course; approval requested in proposal) | `monitor/logistic.py` | Committed (Week 7) |

---

## 3. Team, branches and workflow

### Suggested ownership (team to confirm)

| Area | Lead | Notes |
|---|---|---|
| Environment wrapper, layouts, A*, cooperative planning, scripted saboteur, rollout/logging, integration | Rohan | Week 6 foundation; owns `env/`, `planning/`, `agents/`, `eval/` |
| Monitor: features, dataset builder, threshold rule, logistic regression, quarantine-and-replan | Nathan | Weeks 7 and 9; owns `monitor/` |
| Single-aisle gridworld, value iteration, SARSA vs. Q-learning, DQN | Yiming | Weeks 8 and 9; owns `rl/` |
| Report, figures, GIFs | All | Week 10 |

### Git rules

- `main` is always runnable: `pytest` passes and `scripts/run_baseline.py --smoke` finishes.
- Never commit directly to `main`. Work on a branch, open a pull request, and get one teammate to review before merging.
- Branch names: `feat/<short-topic>` for features, `fix/<topic>` for bugs, `chore/<topic>` for setup, `exp/<topic>` for throwaway experiments.
- Keep branches short-lived (a few days). Rebase on `main` before opening the PR: `git fetch origin && git rebase origin/main`.
- Small, focused commits with imperative messages ("Add space-time A* with reservation table").
- Do not commit: raw episode logs, model checkpoints, large GIFs, `.ipynb_checkpoints`, `__pycache__`, `.DS_Store`. Small summary CSVs and at most a few GIFs under 3 MB (in `docs/media/`) are fine.
- Shared interfaces (section 6) change only through a PR that all three people see.

### Environment

- Conda environment name: `cogs188`, Python 3.11, created from `environment.yml`, which installs `requirements.txt` via pip.
- After creating it: `pip install -e .` so `import warehouse_sabotage` works everywhere (scripts, tests, notebooks).
- Register the kernel for notebooks: `python -m ipykernel install --user --name cogs188`.
- If a new dependency is needed, add it to `requirements.txt` in the same PR and say so in the PR description.

### Dependencies (`requirements.txt`)

Verified to install together on 2026-11-07 (Python 3.13 test; target is 3.11). rware is pinned because we read its internals.

```
rware==2.0.0
gymnasium>=1.0
numpy>=1.26
networkx>=3.2
pyglet<2          # rware renderer requires pyglet 1.x
six               # rware renderer imports six; missing by default
pandas>=2.2
pyarrow>=15       # parquet episode logs
scikit-learn>=1.4
torch>=2.2
matplotlib>=3.8
imageio>=2.34
pyyaml>=6.0
tqdm>=4.66
pytest>=8.0
ipykernel>=6.29
```

---

## 4. rware facts (verified against rware 2.0.0 source)

Do not rely on memory of older rware versions. These were checked by reading `rware/warehouse.py` and running the env.

**Creating the env**
- `import rware` registers ids like `rware-tiny-4ag-v2` (sizes tiny/small/medium/large, 1-19 agents, optional `-easy`/`-hard`).
- Defaults for registered ids: `sensor_range=1`, `max_steps=500`, `request_queue_size = n_agents` (4 for 4ag), `reward_type=RewardType.INDIVIDUAL`, `msg_bits=0`, observation type FLATTENED.
- Any constructor kwarg can be overridden: `gym.make("rware-tiny-4ag-v2", layout=LAYOUT_STR, max_steps=500)`.
- Custom layouts are strings of `x` (shelf slot), `.` (highway) and `g` (goal, also a highway). Must be rectangular with at least one goal.

**Interface**
- `reset(seed=...)` returns `(obs_tuple, info)`. Each obs is a flat float vector (71 dims for tiny, sensor_range 1).
- `step(actions)` takes a list/tuple of 4 ints and returns `(obs_tuple, rewards_list, done, truncated, info)`. `truncated` is always `False`; `done` becomes `True` at `max_steps`. `info` is empty.
- Actions: `0 NOOP, 1 FORWARD, 2 LEFT, 3 RIGHT, 4 TOGGLE_LOAD`. LEFT and RIGHT rotate in place (RIGHT = clockwise in the order UP, RIGHT, DOWN, LEFT). Direction enum: `UP=0 (y-1), DOWN=1 (y+1), LEFT=2 (x-1), RIGHT=3 (x+1)`.
- 500 random steps take about 0.2 s, so the simulator is never the bottleneck.

**State internals** (read through `env.unwrapped`)
- `grid_size = (rows, cols)`; tiny is `(11, 10)`. Positions are `(x=col, y=row)`. Arrays are indexed `[y, x]`: `highways[y, x]`, `grid[layer, y, x]` with layer 0 = agents, layer 1 = shelves.
- IDs in `grid` are 1-indexed (0 = empty); `agents` and `shelfs` lists are 0-indexed. Note the spelling `shelfs`.
- `agents[i]` has `.x, .y, .dir, .carrying_shelf` (a `Shelf` or `None`), `.id`.
- `request_queue` is a list of `Shelf` objects. `goals` is a list of `(x, y)`; tiny goals are `(4, 10)` and `(5, 10)`.
- Robots spawn at random tiles with random facing, possibly underneath shelves.

**Movement rules that matter for planning**
- An *unloaded* robot only collides with robots, so it can drive underneath shelves.
- A *loaded* robot cannot move onto a tile that holds a standing shelf. In practice it travels on highways and empty shelf slots.
- TOGGLE_LOAD while unloaded picks up the shelf under the robot (if any). TOGGLE_LOAD while loaded drops the shelf **only on a non-highway tile**; on a highway it silently does nothing.
- Delivery: if a *requested* shelf is on a goal tile at the end of a step, the robot standing there gets `+1` (INDIVIDUAL reward) and the request is immediately replaced by a random non-requested shelf. The robot is still carrying the shelf and must return it to an empty shelf slot before it can do anything else useful.
- Conflicts: two robots swapping tiles head-on both fail; a chain of robots following each other moves; a blocked FORWARD silently becomes NOOP. Planners must therefore check whether a move actually happened.

**Rendering**
- rware's `render` ignores Gymnasium's `render_mode`. Use `env.unwrapped.render(mode="rgb_array")` to get a frame.
- Needs `pyglet<2`, `six` and a display. Works on macOS (a window may open). On headless Linux use `xvfb-run -a python ...`. GIF code must be optional so tests never need a display.

**Layouts**
- Stock tiny is open: the middle band x=3..6 is a 4-wide highway, and the only 1-wide corridors are the edge columns x=0 and x=9. Its natural sabotage targets are the two goal tiles, their access tiles on row 9, and requested shelves.
- We therefore also use a custom `tiny_choke` layout (same 11x10 size) that fills the middle band with shelves, leaving four 1-wide vertical corridors (x=0, 3, 6, 9) and a 1-high delivery row. Blocking one corridor forces a longer detour but never disconnects the map (verified: the highway graph has no articulation points), which is exactly what the reroute demo needs:

```
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
```

- "Chokepoint" is defined operationally, not as a graph articulation point: a highway tile is a chokepoint if it lies in a corridor of width 1 for loaded robots, or it is a goal or a goal access tile. Compute these in `env/layouts.py` and store them with the layout.

---

## 5. Repository layout

```
Rohan-Nathan-Yiming_COGS188/
  CLAUDE.md                  # imports this file for Claude Code
  project.md                 # this file
  README.md                  # setup + how to run, for humans
  environment.yml
  requirements.txt
  pyproject.toml             # package: warehouse_sabotage (src layout)
  .gitignore
  configs/                   # YAML experiment configs
  src/warehouse_sabotage/
    __init__.py
    types.py                 # shared dataclasses and protocols (section 6)
    env/
      make_env.py            # build env from config (layout, n_agents, seed)
      layouts.py             # layout strings, chokepoint detection
      state.py               # extract WarehouseState snapshot from env.unwrapped
      reward_wrapper.py      # SabotageRewardWrapper + event detection
    planning/
      search.py              # BFS, UCS, greedy best-first, A* (one generic best-first core)
      heuristics.py          # Manhattan, Manhattan + turn cost
      cooperative.py         # reservation table, space-time A* with WAIT
      grid_graph.py          # passability for loaded/unloaded robots
    agents/
      base.py                # Policy protocol helpers
      honest_astar.py        # task state machine + A* replanning
      task_assignment.py     # which honest robot takes which request
      saboteur_scripted.py   # scripted saboteur behaviours
      simple.py              # random policy, idle/parked policy
      controller.py          # FleetController: state -> joint action
    monitor/                 # Week 7 (Nathan)
      features.py
      dataset.py
      threshold.py
      logistic.py
      quarantine.py
    rl/                      # Week 8 (Yiming)
      gridworld.py
      dp.py                  # value iteration
      tabular.py             # SARSA, Q-learning (+ optional MC control)
      dqn.py
    eval/
      rollout.py             # run one episode, return logs
      logging.py             # per-step rows -> parquet
      metrics.py             # episode and condition metrics
      experiment.py          # run a config across seeds/conditions
    viz/
      gif.py                 # optional rendering to GIF
      plots.py
  scripts/                   # thin CLIs: argparse + config -> eval/experiment.py
  tests/
  notebooks/                 # exploration only; proposal notebook stays at repo root or here
  results/                   # gitignored except results/summary/*.csv
  data/                      # gitignored (episode logs, datasets)
  docs/media/                # small GIFs/figures for the report
```

Rules: library code goes in `src/warehouse_sabotage/`; scripts stay thin; notebooks import from the package and never define core logic.

---

## 6. Shared interfaces (the contract between teammates)

These let the monitor and RL work proceed from logged data without depending on planner internals. Keep them stable.

```python
# types.py (sketch; exact fields may grow, never silently shrink)
@dataclass(frozen=True)
class RobotState:
    robot_id: int            # 0-indexed, matches env.unwrapped.agents order
    x: int; y: int
    direction: int           # rware Direction value 0..3
    carrying_shelf_id: int | None   # 0-indexed shelf id or None

@dataclass(frozen=True)
class WarehouseState:
    t: int
    robots: tuple[RobotState, ...]
    shelf_positions: dict[int, tuple[int, int]]   # shelf_id -> (x, y)
    requested_shelf_ids: tuple[int, ...]
    highways: np.ndarray        # [y, x] uint8 (treat as read-only)
    goals: tuple[tuple[int, int], ...]
    chokepoints: frozenset[tuple[int, int]]

class Policy(Protocol):
    def reset(self, state: WarehouseState, robot_id: int, rng: np.random.Generator) -> None: ...
    def act(self, state: WarehouseState) -> int: ...          # rware action 0..4
    def debug_info(self) -> dict: ...                          # phase, plan length, planning ms, nodes expanded, replans, sabotage_active

class Monitor(Protocol):
    def reset(self, state: WarehouseState) -> None: ...
    def update(self, state: WarehouseState, actions: Sequence[int]) -> None: ...
    def scores(self) -> dict[int, float]: ...                  # robot_id -> P(saboteur)
    def flagged(self) -> frozenset[int]: ...
```

- The `FleetController` owns one `Policy` per robot, an optional shared reservation table for honest robots, and an optional `Monitor`. Flagged robots' current tiles are passed to honest planners as permanent obstacles.
- Honest planners may use full state (centralised fleet manager). Learning agents (DQN) use rware's local observations only. The monitor is a centralised component and may read full state.
- The saboteur's robot id is drawn per episode from the episode RNG, never fixed to 0, so nothing can learn "robot 0 is bad".

---

## 7. Data logging schema

Every rollout writes one parquet file per episode to `data/episodes/<run_name>/ep_<seed>.parquet`, one row per robot per step. This is the raw material for the Week 7 monitor dataset, so get it right in Week 6.

| Column | Type | Meaning |
|---|---|---|
| run_name, condition, layout, saboteur_behaviour | str | Experiment identifiers (`saboteur_behaviour = "none"` when absent) |
| episode_seed | int | Seed passed to `reset` |
| t | int | Timestep (0-based, state before the action) |
| robot_id | int | 0-indexed |
| is_saboteur | bool | Ground-truth label |
| x, y, direction | int | Position and facing at t |
| carrying_shelf_id | int (nullable) | |
| carrying_requested | bool | Carried shelf is currently requested |
| on_highway, on_chokepoint | bool | Tile properties |
| action | int | Action chosen |
| moved | bool | Position actually changed after the step |
| delivered | bool | This robot delivered at this step |
| picked_up, dropped | bool | Load events |
| sabotage_active | bool | Saboteur is executing a sabotage manoeuvre (False for honest robots); ground truth for TTD and window labels |
| task_phase | str | Honest: `to_shelf`, `to_goal`, `return_shelf`, `idle`; saboteur: behaviour-specific phase |
| plan_len, plan_ms, nodes_expanded, replanned | numeric | Planner diagnostics (NaN when not applicable) |

Plus one row per episode appended to `results/<run_name>/episodes.csv` with the episode metrics in section 8.

---

## 8. Metrics (must match the proposal)

**Fleet**
- Throughput `D`: requested shelves delivered by honest robots per episode.
- Throughput retained `R = D(saboteur + monitor) / D(no saboteur)`; in Week 6 (no monitor yet) also report `D(saboteur) / D(no saboteur)`.
- Deadlock rate: fraction of episodes with no delivery for `L` consecutive steps while requests are outstanding. Default `L = 100`; keep it a config value.
- Control condition: "3 honest + 1 parked idle robot" separates the cost of losing one worker from the cost of sabotage.

**Monitor (Week 7 onward)**
- `TTD = t_flag - t_first_sabotage` (from `sabotage_active`); if never flagged, `TTD = H - t_first_sabotage` and report detection rate.
- False-positive rate at robot-episode level; window-level precision, recall, F1, ROC-AUC.

**Learning (Week 8 onward)**: mean return with std across seeds; tabular learned path and deadlock rate.

**Computational**: planning time per call and per environment step, nodes expanded, replans per episode; training steps and wall-clock time.

**Reporting rule**: every number is mean ± std over seeds (at least 5 seeds for learned methods; at least 30 episodes per condition for scripted ones). Seeds are explicit in configs.

---

## 9. Design decisions already made

- **Saboteur rewards** (custom wrapper; used for analysis now and for training the learned saboteur later): `-10` per honest delivery, `+5` when it drops a currently requested shelf at a slot farther (loaded A* distance) from the nearest goal than where it picked it up, `-50` when flagged by the monitor.
- **Honest rewards**: `+10` per delivery, `-0.01` per step; DQN may add shaping (e.g. `+1` for picking up a requested shelf), configurable and logged.
- **Honest task loop**: idle → claim a requested shelf (greedy nearest unclaimed, shared claim set) → drive (unloaded) to it → TOGGLE_LOAD → drive (loaded) to the nearest free goal → deliver → drive (loaded) to the nearest empty shelf slot → TOGGLE_LOAD → idle.
- **A\***: state `(x, y, facing)`, actions FORWARD/LEFT/RIGHT (+ WAIT in space-time search), unit costs. Heuristic: Manhattan distance plus 1 if the goal is not reachable without turning; it must be admissible and consistent, and tests must prove it on small grids by comparing against UCS.
- **Cooperative planning**: prioritised space-time A* with a reservation table (vertex and edge-swap conflicts), finite horizon (default 32 steps) then fall back to spatial A*. The saboteur is not in the reservation table; honest robots only see its current position, which is why it can break them.
- **Replanning triggers**: path blocked, move failed, reservation conflict, or every `K` steps (default 8). A stuck counter triggers a randomised yield to break honest-honest deadlocks.
- **Scripted saboteur behaviours** (selectable, logged as `saboteur_behaviour`):
  1. `goal_blocker`: loiters near the delivery area looking busy and steps onto a goal or goal access tile when a loaded honest robot is within `d` steps.
  2. `corridor_parker`: waits near a chokepoint corridor and enters it when an honest robot's planned path uses it.
  3. `shelf_squatter`: parks underneath a requested shelf that an honest robot has claimed.
  4. `shelf_hider`: picks up a requested shelf and drops it at the farthest empty slot from the goals.
  All behaviours share a `camouflage` probability of acting like an honest robot on a given step, so sabotage is intermittent and covert. Week 6 needs at least `goal_blocker`, `corridor_parker` and `shelf_hider`.

---

## 10. Roadmap

### Week 6 (Phase 1, Rohan): environment, A* baseline, scripted saboteur
Deliverables: scaffold and conda env; layouts and chokepoints; state extraction; reward wrapper with events; BFS/UCS/greedy/A* with planner comparison; cooperative A* fleet; scripted saboteur behaviours; rollout logging; metrics; baseline experiment across conditions (`no_saboteur`, `idle_robot`, each saboteur behaviour; stock tiny and `tiny_choke`); first GIFs; short results note.
Acceptance: `pytest` green; `run_baseline.py --smoke` under 1 minute; no-saboteur deliveries clearly above the random-policy baseline and low deadlock rate (if not, fix the planner before interpreting sabotage results); saboteur conditions show a measurable throughput drop relative to the idle control.

### Week 7 (Phase 2, Nathan): monitor
Feature extraction over sliding windows (12 features listed in the proposal), dataset builder from episode parquet files with episode-grouped splits, threshold monitor, logistic regression (class weighting, `GroupKFold`), quarantine-and-replan integration, reroute demo GIF.

### Week 8 (Phase 3, Yiming): tabular and DQN
Single-aisle gridworld MDP with a stochastic blocker; value iteration ground truth; SARSA vs. Q-learning (gamma sweep, two epsilon schedules); optional MC control; Double/Dueling DQN controlling one honest robot (others A*); move experiments to the small map.

### Week 9 (Phase 4, all): selection and evaluation
Feature ablation; window length and threshold search with hill climbing vs. simulated annealing; limited RL hyperparameter grid across seeds; DQN vs. A*; full evaluation table; learned saboteur extension if time allows; hold-one-behaviour-out test of monitor generalisation.

### Week 10 (Phase 5, all): report
Figures, side-by-side GIFs, final report PDF with background, methods, results, conclusion and repo link.

---

## 11. Coding conventions

- Python 3.11, type hints, docstrings that name the course concept implemented.
- All randomness goes through `numpy.random.Generator` objects created from config seeds. No global `np.random` calls.
- Configs in YAML; scripts accept `--config` and `--smoke` (tiny run for CI-style checks).
- Tests in `tests/` with pytest; tests must not require a display or GPU and should run in under 30 s total.
- Prefer small pure functions for planning and feature code so they can be tested on hand-made grids.
- Log, do not print, inside library code (`logging` module).

## 12. Writing conventions for report text, README and results notes

British spelling (behaviour, optimise, colour). No em dashes. Plain, specific language; no filler. Respect word limits exactly when one is given.

## 13. Open questions (decide and record here)

- Final value of `L` for deadlock and window/stride for features (proposal suggests about 20 and 10).
- Whether the honest DQN robot uses rware's flat observation or an image observation.
- Exact "small" map layout for Weeks 8-9 (stock small vs. a choke variant).
