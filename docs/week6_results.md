# Week 6 results: A* fleet baseline and scripted saboteur

Setup: rware tiny map (11 x 10), 4 robots, 500 steps per episode, 30 episodes per condition on the same explicit seeds (1000 to 1029), so every condition sees the same spawns and initial requests. No monitor yet. Config: `configs/week6_baseline.yaml`. Reproduce with `python scripts/run_baseline.py` (about 2 minutes on a laptop). Raw numbers: `results/summary/week6_baseline.csv` and `results/summary/planner_comparison.csv`.

Conditions: `random_policy` (all four robots random), `no_saboteur` (four honest A* robots), `idle_robot` (three honest robots plus one parked under a non-requested shelf, the control for losing one worker), and one condition per saboteur behaviour (three honest robots plus one saboteur, camouflage 0.2). Deliveries D count honest deliveries only. Values are mean ± std over 30 episodes.

## Acceptance criteria

| Criterion | Result |
|---|---|
| `pytest` green and under 30 s | Met: 30 tests in about 2 s |
| `run_baseline.py --smoke` under 1 minute | Met: under 10 s |
| No-saboteur deliveries clearly above the random policy, low deadlock rate | Met: 64.7 against 0.2 deliveries on tiny; deadlock rate 0.10 and 0.07 (by construction of L, see below) |
| The main saboteur behaviours (goal_blocker, corridor_parker, shelf_squatter) are measurably below the idle control | Met on both layouts (below idle in 97 to 100% of paired seeds, p < 1e-11) |

shelf_hider is reported separately as a finding rather than an acceptance criterion (see below).

## Deadlock threshold L

L was chosen after seeing the Week 6 data, not in advance. For each layout it is the 95th percentile (rounded) of the longest no-delivery gap per episode in the `no_saboteur` condition: **L = 26 on stock tiny and L = 30 on tiny_choke**. A deadlock is a gap of at least L steps, and a few no-saboteur episodes sit exactly at the threshold, so the no-saboteur deadlock rate is 0.10 and 0.07 rather than exactly 0.05. With the earlier default of L = 100 no A* episode deadlocked (longest gap 89 steps), so that value could not separate conditions. Because L was tuned on this data, deadlock rates here are descriptive; future weeks should keep these values fixed.

## Fleet results

| Condition | Layout | Deliveries D | D / D(no saboteur) | Longest gap (steps) | Deadlock rate | Plan ms per call | Plan ms per step | Nodes per call | Replans per episode |
|---|---|---|---|---|---|---|---|---|---|
| random_policy | tiny | 0.2 ± 0.4 | 0.00 | 472 ± 66 | 1.00 | - | 0.00 | - | 0 ± 0 |
| random_policy | tiny_choke | 0.1 ± 0.3 | 0.00 | 482 ± 55 | 1.00 | - | 0.00 | - | 0 ± 0 |
| no_saboteur | tiny | 64.7 ± 2.1 | 1.00 | 22 ± 3 | 0.10 | 0.08 | 0.05 | 32 | 131 ± 7 |
| no_saboteur | tiny_choke | 56.1 ± 2.4 | 1.00 | 24 ± 3 | 0.07 | 0.07 | 0.05 | 31 | 160 ± 11 |
| idle_robot | tiny | 49.8 ± 1.4 | 0.77 | 25 ± 4 | 0.33 | 0.08 | 0.04 | 32 | 94 ± 5 |
| idle_robot | tiny_choke | 44.9 ± 1.8 | 0.80 | 26 ± 3 | 0.17 | 0.07 | 0.03 | 27 | 109 ± 7 |
| goal_blocker | tiny | 40.9 ± 1.8 | 0.63 | 32 ± 4 | 1.00 | 0.06 | 0.07 | 23 | 101 ± 11 |
| goal_blocker | tiny_choke | 26.2 ± 2.9 | 0.47 | 53 ± 12 | 1.00 | 0.20 | 0.26 | 97 | 192 ± 20 |
| corridor_parker | tiny | 40.9 ± 3.6 | 0.63 | 32 ± 6 | 0.97 | 1.10 | 1.19 | 519 | 166 ± 21 |
| corridor_parker | tiny_choke | 30.7 ± 3.0 | 0.55 | 44 ± 6 | 1.00 | 0.60 | 0.65 | 293 | 193 ± 16 |
| shelf_hider | tiny | 48.3 ± 2.0 | 0.75 | 25 ± 4 | 0.40 | 0.14 | 0.07 | 61 | 97 ± 5 |
| shelf_hider | tiny_choke | 44.0 ± 2.1 | 0.78 | 28 ± 4 | 0.43 | 0.11 | 0.06 | 51 | 109 ± 6 |
| shelf_squatter | tiny | 45.6 ± 2.0 | 0.70 | 27 ± 4 | 0.63 | 0.13 | 0.12 | 58 | 114 ± 13 |
| shelf_squatter | tiny_choke | 36.3 ± 2.5 | 0.65 | 36 ± 6 | 0.87 | 0.20 | 0.18 | 95 | 142 ± 16 |

Paired comparison with the idle-robot control (same seeds, saboteur minus idle deliveries, paired t-test):

| Behaviour | tiny | tiny_choke |
|---|---|---|
| goal_blocker | -9.0 ± 2.4 (below in 100% of seeds, p < 1e-17) | -18.6 ± 3.4 (100%, p < 1e-22) |
| corridor_parker | -9.0 ± 3.6 (97%, p < 1e-13) | -14.2 ± 3.2 (100%, p < 1e-20) |
| shelf_squatter | -4.3 ± 2.0 (100%, p < 1e-11) | -8.6 ± 2.6 (100%, p < 1e-16) |
| shelf_hider | -1.5 ± 2.2 (73%, p = 6e-4) | -0.9 ± 2.5 (57%, p = 0.055) |

## Finding: shelf_hider is subtle and low-harm on tiny maps

shelf_hider picks up about 16.5 requested shelves per episode and drops each one at the empty slot farthest from the goals, yet it costs only 1.5 deliveries against the idle control on stock tiny (about 3%) and is not measurably worse than the idle control on tiny_choke (p = 0.055). On an 11 x 10 map every slot is close to the goals, so the farthest slot is only a few steps farther than where the shelf started and honest robots recover quickly. We have not tuned it to pass. It stays in the experiment as a subtle, low-harm behaviour (useful as a hard case for the monitor) and will be re-tested on the small map in Week 8, where distances to the goals are longer.

## Planner comparison

100 random start/goal pairs per layout and load state, on the reset-time map (loaded robots restricted to highways). Mean nodes expanded:

| Layout | Load | BFS | UCS | Greedy (share optimal) | A* |
|---|---|---|---|---|---|
| tiny | unloaded | 169 | 202 | 10 (80%) | 10 |
| tiny | loaded | 131 | 152 | 21 (80%) | 21 |
| tiny_choke | unloaded | 172 | 209 | 10 (73%) | 10 |
| tiny_choke | loaded | 82 | 95 | 17 (87%) | 17 |

A* always matched the optimal UCS cost and expanded 5 to 20 times fewer nodes. Greedy best-first expanded about as few nodes as A* but returned a longer path on 13 to 27% of pairs. Every search took under 0.3 ms on average.

## Observations

1. The A* fleet works: 64.7 deliveries per episode on stock tiny against 0.2 for the random policy, with a longest delivery gap of 22 ± 3 steps. Planning is cheap (under 0.1 ms per call without a saboteur).
2. Removing one honest robot (idle control) costs 20 to 23% of throughput. Saboteurs that block space cost far more: goal_blocker on tiny_choke keeps only 47% of normal throughput, is 18.6 deliveries below the idle control in every seed and more than doubles the longest gap (53 against 24 steps).
3. The choke layout amplifies sabotage, as intended. Every space-blocking behaviour hurts more on tiny_choke than on stock tiny, because there are only two goal access tiles and four 1-wide corridors.
4. corridor_parker is also strong on stock tiny. It parks in the edge columns x=0 and x=9, which honest robots use to reach the outer shelf columns. It also makes planning about 14 times more expensive per call (519 against 32 nodes, 1.1 against 0.08 ms), because space-time A* exhausts its horizon when a corridor is blocked and then falls back to spatial A*.
5. Deadlock rates now separate conditions: with the data-derived L, every goal_blocker and corridor_parker episode has at least one deadlock, against 7 to 10% without a saboteur and 17 to 33% for the idle control.

## Problems found

- shelf_hider is weak on tiny maps (see the finding above).
- Space-time A* sometimes costs up to 30 ms per call when a target is cut off (about 3% of calls under corridor_parker). That is fine on tiny but worth watching on the small map.
- When goal_blocker stops blocking, it often drives back through the goal tiles while labelled `loiter` with `sabotage_active = False`. The label follows the definition, but windows near the goals will be noisy.
- On macOS, rware's renderer raises when its pyglet window is closed. `viz/gif.py` works around this; tests never render.

## What Week 7 needs from the logs

- Episode logs are in `data/episodes/<run_name>/ep_<seed>.parquet` (one row per robot per step, with the agreed schema plus `reward` and `planned_cells` columns), with `run_name = week6_baseline__<condition>__<layout>`. Load them with `warehouse_sabotage.eval.logging.load_run("data", "week6_baseline__goal_blocker__*")`.
- Labels: `is_saboteur` (robot level) and `sabotage_active` (step level) for TTD and window labels. The idle robot is not labelled as a saboteur, which makes it a useful hard negative (it dwells but does not sabotage).
- `planned_cells` holds each honest robot's next (up to) 8 planned tiles as [x, y] pairs, starting with the tile it should occupy at t+1. It is empty for the saboteur, parked and random robots, and for honest robots that are holding still (picking up, dropping or waiting with no path). After reading the parquet each entry is an array of 2-element arrays. Aisle-blocking can be computed as "robot j stands on a tile in robot i's `planned_cells`".
- The other proposal features can be computed from `x`, `y`, `on_chokepoint`, `moved`, `carrying_requested`, `picked_up`, `dropped`, `delivered` and `action`.
- 30 episodes per condition on two layouts gives 60 saboteur and 180 honest robot-episodes per behaviour. For the 500+ episode dataset in the proposal, rerun with more seeds; one episode takes about 0.3 s.
