# Week 7 follow-up: why quarantine hurt against `goal_blocker`, and the fix

## Problem

In Week 7 the logistic monitor (Version 2) detected `goal_blocker` in every episode, yet the monitored fleet delivered fewer shelves than a fleet with no monitor: -10.5 [-14.3, -6.5] on `tiny` and -16.1 [-17.8, -14.2] on `tiny_choke` (paired against the blind fleet). The cause was not tested.

The suspected cause was the quarantine design. When a robot is first flagged, the tile it stands on is blocked for the honest planners **for the rest of the episode**. `goal_blocker` is usually flagged while it stands on a goal or a goal access tile, so one of the two goals or its only access is lost for good. On `tiny_choke`, 32% of honest robot-episodes were also falsely flagged while queuing behind the blocker. Each of those flags blocked another tile next to the goals, permanently.

## What changed

`planner.quarantine_steps` (in `agents/controller.py`) sets how long the tile a robot was flagged on stays blocked:

| value | meaning |
|---|---|
| `0` (new default) | never: the honest fleet avoids only the flagged robot's **current** tile and stops cooperating with it (`ctx.flagged`) |
| `null` | for the rest of the episode (the Week 7 design; `configs/week7_eval*.yaml` now set it explicitly so Week 7 results reproduce) |
| `N` | for N steps after the flag |

An experiment condition can override planner settings (`{kind, oracle, monitor, planner}` in `eval/experiment.py`), and `scripts/paired_analysis.py` reads its comparisons from a `pairs:` list in the config when one is given.

## Experiment

```bash
python scripts/run_baseline.py --config configs/week7_data.yaml                  # regenerates the Week 7 data
python scripts/train_monitor.py --selection episode --suffix _v2                 # same tau 0.860 and persist 4 as Week 7
python scripts/eval_monitor.py --config configs/week7_quarantine.yaml            # selection run, about 30 min
python scripts/paired_analysis.py --config configs/week7_quarantine.yaml
python scripts/eval_monitor.py --config configs/week7_quarantine_confirm.yaml    # confirmation run, about 30 min
python scripts/paired_analysis.py --config configs/week7_quarantine_confirm.yaml
```

- Same Version 2 logistic monitor as Week 7. It was retrained from regenerated data: coefficients match the committed ones to 1e-11, with the same tau and persist.
- Three quarantine modes: `permanent` (Week 7), `off`, and `timed` (50 steps; one value fixed in advance, not tuned). Each was compared with the blind fleet (no monitor) and the oracle fleet (Week 6, knows the saboteur from step 0), for every behaviour, no saboteur and the idle robot, on `tiny` and `tiny_choke`, 30 episodes per cell.
- **Selection run:** seeds 3100 to 3129, the Week 7 Version 2 evaluation seeds. The mode was chosen here. Its `permanent` cells reproduce the Week 7 numbers exactly (for example `goal_blocker` 19.2 and 5.9 deliveries, -10.47 and -16.10 against blind), which checks the regenerated monitor and the code change.
- **Confirmation run:** fresh seeds 3200 to 3229, run once after `off` was chosen. The numbers below are from this run (`results/summary/week7_quarantine_confirm*.csv`); the selection run is in `results/summary/week7_quarantine*.csv`.

## Results (confirmation run, seeds 3200 to 3229)

Honest deliveries per episode, mean (std) over 30 seeds:

| behaviour | layout | oracle | blind | permanent (Week 7) | off (new default) | timed 50 |
|---|---|---|---|---|---|---|
| goal_blocker | tiny | 40.7 (2.0) | 30.4 (3.4) | 13.9 (8.2) | 39.1 (2.6) | 36.5 (2.6) |
| goal_blocker | tiny_choke | 26.6 (1.9) | 22.6 (3.6) | 6.6 (5.9) | 23.0 (4.4) | 22.7 (3.6) |
| corridor_parker | tiny | 39.8 (4.0) | 30.8 (13.2) | 29.2 (11.4) | 39.7 (3.9) | 39.6 (3.6) |
| corridor_parker | tiny_choke | 29.7 (3.4) | 18.7 (9.1) | 20.8 (7.2) | 28.6 (5.8) | 28.0 (5.6) |
| shelf_hider | tiny | 47.4 (1.5) | 40.9 (2.5) | 44.4 (2.3) | 46.4 (1.8) | 46.2 (1.7) |
| shelf_hider | tiny_choke | 42.7 (2.2) | 37.9 (2.3) | 39.4 (3.2) | 42.3 (2.2) | 42.3 (2.0) |
| shelf_squatter | tiny | 45.1 (2.2) | 31.6 (2.3) | 40.8 (2.2) | 43.5 (2.1) | 43.3 (1.8) |
| shelf_squatter | tiny_choke | 37.1 (2.6) | 29.9 (2.3) | 34.3 (3.2) | 35.9 (2.9) | 35.8 (3.5) |
| no saboteur | tiny | | 64.5 (2.1) | 64.3 (2.4) | 64.4 (2.2) | 64.3 (2.4) |
| no saboteur | tiny_choke | | 56.2 (3.0) | 56.2 (3.0) | 56.2 (3.0) | 56.2 (3.0) |
| idle robot | tiny | | 47.6 (2.3) | 47.7 (2.2) | 47.7 (2.2) | 47.7 (2.1) |
| idle robot | tiny_choke | | 44.2 (2.3) | 44.3 (2.2) | 44.3 (2.1) | 44.3 (2.1) |

"Blind" is the fleet without a monitor; in the no-saboteur rows it is the plain honest fleet.

Paired differences, monitored fleet minus blind fleet (same seeds, bootstrap 95% interval, 10,000 resamples):

| behaviour | layout | permanent (Week 7) | off (new default) |
|---|---|---|---|
| goal_blocker | tiny | -16.5 [-19.2, -13.4] | **+8.7 [7.3, 10.0]** |
| goal_blocker | tiny_choke | -15.9 [-18.1, -13.4] | +0.4 [-1.3, 2.1] |
| corridor_parker | tiny | -1.6 [-5.7, 2.3] | **+8.9 [5.2, 13.1]** |
| corridor_parker | tiny_choke | +2.1 [-1.1, 5.3] | **+9.9 [7.0, 12.9]** |
| shelf_hider | tiny | +3.5 [2.4, 4.5] | **+5.5 [4.4, 6.6]** |
| shelf_hider | tiny_choke | +1.5 [0.4, 2.7] | **+4.4 [3.5, 5.3]** |
| shelf_squatter | tiny | +9.2 [8.3, 10.1] | **+11.8 [11.0, 12.7]** |
| shelf_squatter | tiny_choke | +4.4 [3.6, 5.3] | **+6.0 [5.2, 6.8]** |
| no saboteur | tiny / tiny_choke | -0.2 / 0.0 | -0.1 / 0.0 |
| idle robot | tiny / tiny_choke | 0.0 / 0.0 | +0.1 / +0.1 |

`off` minus `permanent` is positive with an interval excluding 0 in all 8 saboteur cells: +25.2 and +16.4 for `goal_blocker`, +10.5 and +7.8 for `corridor_parker`, +2.0 and +2.9 for `shelf_hider`, +2.6 and +1.6 for `shelf_squatter`. Detection is unchanged (the monitor is the same; the quarantine only changes what the fleet does after a flag): detection rate 0.97 to 1.00 and TTD 58 to 145 steps.

## Findings

- **The permanent quarantine caused the `goal_blocker` failure.** Removing it turns the loss into a gain on `tiny` (+8.7) and removes it on `tiny_choke` (+0.4, interval includes 0). The selection run showed the same pattern (+9.6 and +2.3).
- **Quarantine off is the best mode for every behaviour,** on both seed ranges. It also makes the monitor useful against `corridor_parker` (+8.9 and +9.9), where the permanent version had no measurable effect. With it, the monitored fleet comes within 1.6 deliveries of the oracle fleet in 7 of the 8 saboteur cells (all but `goal_blocker` on `tiny_choke`), despite detecting 58 to 145 steps late.
- **The timed quarantine (50 steps) sits between the two** and is never better than off, so we drop it.
- **No cost when there is no saboteur.** On these seeds the monitor rarely flags honest robots without a blocker (FPR 0.00 to 0.05), and without a permanent block a false flag costs almost nothing (-0.1 to +0.1 deliveries).
- **What is left: `goal_blocker` on `tiny_choke`.** Off only matches the blind fleet there (23.0 against 22.6; oracle 26.6). The likely reason is false flags: 44% of honest robot-episodes get flagged while queued behind the blocker. A flagged honest robot is treated as an obstacle and dropped from the reservation table, which slows the queue at the only delivery row. Making the monitor robust to queuing (for example, training data with more queuing, or ignoring flags on robots in a delivery queue) is a Week 9 item, not a quarantine question.

## Deviation from the proposal

The proposal says the flagged robot's tile becomes a permanent obstacle so the honest fleet reroutes. We keep the reroute: the flagged robot is always avoided at its current tile and dropped from cooperative planning. We drop the *permanent* block of the tile where it was flagged, because this experiment shows that it costs throughput in every saboteur condition. The report should state this change and cite this note.

## Limitations

- One monitor (Version 2 logistic) and one timed value (50 steps). Other monitors or durations were not tried.
- Intervals cover 30 seeds per cell, with no correction for the many comparisons.
- The mode was chosen on seeds the Week 7 evaluation had already used. The fresh-seed run confirms the choice, but it was not pre-registered.
