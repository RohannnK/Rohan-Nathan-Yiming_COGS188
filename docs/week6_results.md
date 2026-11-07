# Week 6 results: A* fleet baseline and scripted saboteur

Setup: rware tiny map (11 x 10), 4 robots, 500 steps per episode, 30 episodes per condition on the same explicit seeds (1000 to 1029), so every condition sees the same spawns and initial requests. No monitor yet. Config: `configs/week6_baseline.yaml`. Reproduce with `python scripts/run_baseline.py` (about 2 minutes on a laptop). Raw numbers: `results/summary/week6_baseline.csv` and `results/summary/planner_comparison.csv`.

Conditions: `random_policy` (all four robots random), `no_saboteur` (four honest A* robots), `idle_robot` (three honest robots plus one parked under a non-requested shelf, the control for losing one worker), and one condition per saboteur behaviour (three honest robots plus one saboteur, camouflage 0.2). Deliveries D count honest deliveries only. Values are mean ± std over 30 episodes.

## Fleet results

| Condition | Layout | Deliveries D | D / D(no saboteur) | Deadlock rate | Plan ms per call | Plan ms per step | Nodes per call | Replans per episode |
|---|---|---|---|---|---|---|---|---|
| random_policy | tiny | 0.2 ± 0.4 | 0.00 | 1.00 | - | 0.00 | - | 0 ± 0 |
| random_policy | tiny_choke | 0.1 ± 0.3 | 0.00 | 1.00 | - | 0.00 | - | 0 ± 0 |
| no_saboteur | tiny | 64.7 ± 2.1 | 1.00 | 0.00 | 0.08 | 0.05 | 32 | 131 ± 7 |
| no_saboteur | tiny_choke | 56.1 ± 2.4 | 1.00 | 0.00 | 0.07 | 0.05 | 31 | 160 ± 11 |
| idle_robot | tiny | 49.8 ± 1.4 | 0.77 | 0.00 | 0.08 | 0.04 | 32 | 94 ± 5 |
| idle_robot | tiny_choke | 44.9 ± 1.8 | 0.80 | 0.00 | 0.07 | 0.03 | 27 | 109 ± 7 |
| goal_blocker | tiny | 40.9 ± 1.8 | 0.63 | 0.00 | 0.06 | 0.07 | 23 | 101 ± 11 |
| goal_blocker | tiny_choke | 26.2 ± 2.9 | 0.47 | 0.00 | 0.20 | 0.26 | 97 | 192 ± 20 |
| corridor_parker | tiny | 40.9 ± 3.6 | 0.63 | 0.00 | 1.09 | 1.18 | 519 | 166 ± 21 |
| corridor_parker | tiny_choke | 30.7 ± 3.0 | 0.55 | 0.00 | 0.59 | 0.65 | 293 | 193 ± 16 |
| shelf_hider | tiny | 48.3 ± 2.0 | 0.75 | 0.00 | 0.14 | 0.07 | 61 | 97 ± 5 |
| shelf_hider | tiny_choke | 44.0 ± 2.1 | 0.78 | 0.00 | 0.11 | 0.06 | 51 | 109 ± 6 |
| shelf_squatter | tiny | 45.6 ± 2.0 | 0.70 | 0.00 | 0.13 | 0.12 | 58 | 114 ± 13 |
| shelf_squatter | tiny_choke | 36.3 ± 2.5 | 0.65 | 0.00 | 0.20 | 0.18 | 95 | 142 ± 16 |

Paired comparison with the idle-robot control (same seeds, saboteur minus idle, paired t-test):

| Behaviour | tiny | tiny_choke |
|---|---|---|
| goal_blocker | -9.0 ± 2.4 (below in 100% of seeds, p < 1e-17) | -18.6 ± 3.4 (100%, p < 1e-22) |
| corridor_parker | -9.0 ± 3.6 (97%, p < 1e-13) | -14.2 ± 3.2 (100%, p < 1e-20) |
| shelf_squatter | -4.3 ± 2.0 (100%, p < 1e-11) | -8.6 ± 2.6 (100%, p < 1e-16) |
| shelf_hider | -1.5 ± 2.2 (73%, p = 6e-4) | **-0.9 ± 2.5 (57%, p = 0.055)** |

## Planner comparison

100 random start/goal pairs per layout and load state, on the reset-time map (loaded robots restricted to highways).

| Layout | Load | BFS nodes | UCS nodes | Greedy nodes (optimal) | A* nodes |
|---|---|---|---|---|---|
| tiny | unloaded | 169 | 202 | 10 (80%) | 10 |
| tiny | loaded | 131 | 152 | 21 (80%) | 21 |
| tiny_choke | unloaded | 172 | 209 | 10 (73%) | 10 |
| tiny_choke | loaded | 82 | 95 | 17 (87%) | 17 |

A* always matched the optimal UCS cost and expanded 5 to 20 times fewer nodes. Greedy best-first expanded about as few nodes as A* but returned a longer path on 13 to 27% of pairs. Every search took under 0.3 ms on average.

## Observations

1. The A* fleet works: 64.7 deliveries per episode on stock tiny against 0.2 for the random policy, with no deadlocks and a longest delivery gap of 89 steps across all non-random episodes. Planning is cheap (under 0.1 ms per call without a saboteur).
2. Removing one honest robot (idle control) costs 20 to 23% of throughput. Saboteurs that block space cost far more: goal_blocker on tiny_choke keeps only 47% of normal throughput, and is 18.6 deliveries below the idle control in every seed.
3. The choke layout amplifies sabotage, as intended. Every space-blocking behaviour hurts more on tiny_choke than on stock tiny, because there are only two goal access tiles and four 1-wide corridors.
4. corridor_parker is also strong on stock tiny. It parks in the edge columns x=0 and x=9, which honest robots use to reach the outer shelf columns. It also makes planning about 14 times more expensive per call (519 against 32 nodes, 1.1 against 0.08 ms), because space-time A* exhausts its horizon when a corridor is blocked and then falls back to spatial A*.
5. shelf_hider barely hurts. It picks up about 16.5 requested shelves per episode, but on an 11 x 10 map the farthest empty slot is only a few steps farther from the goals than where the shelf started, so honest robots recover quickly.

## Problems found

- **Acceptance criterion partly failed:** shelf_hider is not measurably below the idle control on tiny_choke (p = 0.055). It is significantly below on stock tiny, but only by about 3%. I have not tuned it to pass. It may become meaningful on the small map (Week 8), or with a variant that hides the shelf far away and then parks on it.
- The deadlock rate is 0 for every A* condition at L = 100; the longest gap under sabotage was 89 steps. L = 100 may be too lenient to separate conditions, and a smaller L (for example 50) or reporting the longest gap directly may be more informative.
- Space-time A* sometimes costs up to 30 ms per call when a target is cut off (about 3% of calls under corridor_parker). That is fine on tiny but worth watching on the small map.
- When goal_blocker stops blocking, it often drives back through the goal tiles while labelled `loiter` with `sabotage_active = False`. The label follows the definition, but windows near the goals will be noisy.
- On macOS, rware's renderer raises when its pyglet window is closed. `viz/gif.py` works around this; tests never render.

## What Week 7 needs from the logs

- Episode logs are in `data/episodes/<run_name>/ep_<seed>.parquet` (one row per robot per step, schema in project.md section 7, plus a `reward` column), with `run_name = week6_baseline__<condition>__<layout>`. Load them with `warehouse_sabotage.eval.logging.load_run("data", "week6_baseline__goal_blocker__*")`.
- Labels: `is_saboteur` (robot level) and `sabotage_active` (step level) for TTD and window labels. The idle robot is not labelled as a saboteur, which makes it a useful hard negative (it dwells but does not sabotage).
- The features in the proposal can be computed from `x`, `y`, `on_chokepoint`, `moved`, `carrying_requested`, `picked_up`, `dropped`, `delivered` and `action`. Path efficiency and aisle-blocking need the honest planned paths, which are not logged yet. If the monitor needs them, we should add a `planned_cells` column next week.
- 30 episodes per condition on two layouts gives 60 saboteur and 180 honest robot-episodes per behaviour. For the 500+ episode dataset in the proposal, rerun with more seeds; one episode takes about 0.3 s.
