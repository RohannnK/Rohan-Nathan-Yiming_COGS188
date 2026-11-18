# Week 7: runtime monitor and quarantine-and-replan

Status: all seven Week 7 deliverables are implemented on branch `feat/week7-monitor` and were run on the real simulator (rware 2.0.0, conda environment `cogs188`). The monitor was evaluated twice. **Version 1** chose its threshold from the window false-positive rate and made the fleet worse, because it flagged most honest robots. **Version 2** chooses the threshold and the number of consecutive flagged windows from the robot-episode false-positive rate. Both versions are reported below and the Version 1 files are kept.

Headline, in plain terms: the logistic monitor detects the saboteur in nearly every episode in both versions, but whether quarantining it helps depends on the behaviour. Version 2 helps against `shelf_squatter` and `shelf_hider`, changes nothing measurable against `corridor_parker` and the idle robot, and **still hurts against `goal_blocker`**. The simple threshold rule never became useful.

## What was implemented

| Piece | File | What it does |
|---|---|---|
| Features | `monitor/features.py` | The 12 proposal features over windows of 20 steps, stride 10 (definitions in the module docstring) |
| Dataset | `monitor/dataset.py` | Window rows from episode logs, robot-level label `is_saboteur`, window-level label `sabotage_window`, split by episode seed |
| Threshold monitor | `monitor/threshold.py` | Flag a window when `choke_steps` or `idle_ticks` reaches a limit; limits chosen on the validation split |
| Logistic regression | `monitor/logistic.py` | StandardScaler + balanced `LogisticRegression`, C by `GroupKFold` on the training split, tau chosen on the validation split |
| Quarantine | `monitor/quarantine.py` | `QuarantineMonitor` (Monitor protocol): scores the live episode with the same feature code, flags permanently, `save_monitor` / `load_monitor` |
| Metrics | `eval/metrics.py` | Window precision, recall, F1, ROC-AUC; TTD, detection rate, robot-episode FPR (offline and closed loop) |
| Episode-level selection (V2) | `monitor/selection.py` | Replays windows in time order with the persistence rule and picks tau and persist by robot-episode FPR on the validation split |
| Scripts, configs | `scripts/train_monitor.py`, `scripts/eval_monitor.py`, `configs/week7_data.yaml`, `configs/week7_eval.yaml`, `configs/week7_eval_v2.yaml` | Data generation reuses `run_baseline.py`; `--selection episode --suffix _v2` trains Version 2 |

## Decisions worth explaining

1. **Oracle versus blind fleet.** In Week 6 `build_fleet` excludes the saboteur from `honest_ids`, so the honest planners treated it as an obstacle from step 0, which means a monitor could not add anything. `FleetContext.oracle` (default `True`, Week 6 behaviour unchanged) now says whether the planners know. With `oracle=False` the saboteur is treated as a cooperating robot until the monitor flags it. The evaluation compares three fleets per behaviour: `_oracle` (Week 6, an upper bound), `_blind` (no monitor, the lower bound) and `_threshold` / `_logistic` (blind until flagged).
2. **Quarantine.** When a robot is first flagged, the tile it stands on is blocked for the honest planners for the rest of the episode (`FleetContext.quarantine`), as in the proposal. The flagged robot's current tile is blocked as well, through the existing `ctx.flagged` hook. The test `test_detection_changes_the_honest_route` checks that the planned route changes once a robot is flagged.
3. **Same code online and offline.** `QuarantineMonitor` builds log-schema rows from `WarehouseState` and the actions, and calls the same `window_table` as the dataset builder. `test_online_features_match_the_logged_features` compares both on a scripted episode and requires identical features.
4. **Leakage.** Features never read `is_saboteur`, `sabotage_active`, `task_phase` or planner diagnostics (`test_features_ignore_label_columns`). Seeds are shared by every condition and layout, so episodes with the same seed have the same spawns, requests and saboteur id; the split groups by `episode_seed`, not by run. The scaler is fitted on the training windows only. Thresholds and tau are chosen on the validation split, the test split is used once for reporting. Closed-loop evaluation uses seeds 3000 to 3029, which never touch training.
5. **Two assumptions to state in the report.** Robots report their planned tiles and current target to the fleet manager (the monitor reads them from the controller's `debug_info`, the same values the logs hold). Feature 7 counts any other robot that stopped on a highway tile for 3 steps, since the monitor cannot know who is honest.
6. **Feature definitions the proposal leaves open.** Corridor tiles are chokepoints that are not goal or goal access tiles, so `corridor_frac` and `choke_steps` differ; `requested_undelivered` is requested pick-ups minus deliveries in the window (at least 0); `path_efficiency` is the shortest-path distance covered towards the target held at the window start, per step (0 without a target).

## Results

Data: 720 episodes (`configs/week7_data.yaml`, seeds 2000 to 2059, 12 condition and layout runs of 60 episodes), 138,240 windows, split by seed into 36 train, 12 validation and 12 test seeds. The share of rows without a planner target was 0.0106 for honest robots and 0.0114 for saboteurs, so `path_efficiency` does not leak the label. Closed-loop Version 1 used seeds 3000 to 3029, Version 2 seeds 3100 to 3129 (30 episodes per cell). Numbers come from `results/summary/week7_monitor.csv`, `week7_monitor_v2.csv`, `week7_eval*.csv` and `week7_eval*_monitor.csv`.

### Version 1 (window false-positive rate at most 5%, persist 1)

tau was the lowest threshold with a window FPR of at most 0.05 on the validation split. The threshold rule found no pair of limits with any recall at that FPR, so both limits ended at 21, above the window length of 20: it never fires and behaves exactly like the blind fleet. The logistic monitor separated windows well (test ROC-AUC 0.978) but flagged **62.7%** of honest robot-episodes on the test split, because one robot has about 48 windows per episode and a 5% window FPR compounds. Quarantine blocks the flagged tile for the rest of the episode, so false flags are expensive, above all on `tiny_choke`: with no saboteur at all the fleet delivered 15.3 per episode instead of 56.0 and deadlocked in every episode.

### Version 2 (robot-episode false-positive rate at most 5%, chosen on the validation split)

Each candidate threshold and each persist value (1, 2, 3, 4, 6) is replayed over every robot's windows in time order, as `QuarantineMonitor` does online, and judged by the share of honest robot-episodes flagged at least once. Among the candidates within 5% the one with the highest detection rate wins. This used only the validation split. The seeds 3000 to 3029 were not used. Version 2 was nevertheless designed after seeing the Version 1 closed-loop results, so seeds 3100 to 3129 make it an independent re-check, not a second blind test, and no further tuning was done on them.

Chosen settings: logistic tau 0.860 with persist 4 (129 of 495 validation candidates met the FPR limit); threshold limits 18 and 21 with persist 6 (8 of 2420 met it, the best detecting 41% of saboteurs on validation). The 50% detection level the training script warns about is a diagnostic of ours, not a project requirement; the threshold rule fell below it.

### Offline monitor performance on the held-out test split (12 seeds, 96 saboteur robot-episodes, 480 honest robot-episodes)

| Monitor | persist | Window ROC-AUC | Window precision | Window recall | Window F1 | Detection rate | TTD mean (std) | Robot-episode FPR |
|---|---|---|---|---|---|---|---|---|
| V1 threshold | 1 | 0.500 | 0.000 | 0.000 | 0.000 | 0.00 | 494.4 (7.6) | 0.000 |
| V1 logistic | 1 | 0.978 | 0.742 | 0.924 | 0.823 | 1.00 | 23.0 (17.2) | 0.627 |
| V2 threshold | 6 | 0.634 | 0.537 | 0.320 | 0.401 | 0.47 | 334.4 (182.2) | 0.081 |
| V2 logistic | 4 | 0.978 | 0.828 | 0.793 | 0.810 | 1.00 | 91.7 (57.6) | 0.042 |

### Throughput (honest deliveries per episode, mean and standard deviation over 30 seeds)

The last column pairs every episode with the blind fleet on the same seed (`scripts/paired_analysis.py`, seeded bootstrap, 10,000 resamples; all 44 condition and layout runs ran the full 500 steps). V1 columns use seeds 3000 to 3029, so the blind fleet they were compared with differs slightly.

| Behaviour | Layout | Oracle (Week 6) | Blind | V1 logistic | V2 threshold | V2 logistic | V2 logistic minus blind, paired, 95% bootstrap interval |
|---|---|---|---|---|---|---|---|
| goal_blocker | tiny | 40.3 (2.0) | 29.7 (3.2) | 11.2 vs blind 30.4 | 19.4 (10.0) | 19.2 (11.5) | -10.5 [-14.3, -6.5] |
| goal_blocker | tiny_choke | 26.2 (3.2) | 22.0 (3.4) | 3.7 vs blind 21.1 | 9.7 (6.2) | 5.9 (4.8) | -16.1 [-17.8, -14.2] |
| corridor_parker | tiny | 39.5 (3.1) | 33.7 (9.2) | 26.9 vs blind 32.9 | 34.6 (8.3) | 31.7 (9.5) | -2.0 [-6.2, 2.2] |
| corridor_parker | tiny_choke | 30.0 (3.2) | 20.2 (7.7) | 12.9 vs blind 19.5 | 20.6 (6.9) | 20.1 (6.3) | -0.1 [-3.8, 3.4] |
| shelf_hider | tiny | 47.4 (1.8) | 40.9 (2.8) | 40.8 vs blind 41.6 | 40.9 (2.8) | 43.7 (2.5) | +2.9 [1.6, 4.0] |
| shelf_hider | tiny_choke | 42.6 (1.9) | 37.7 (2.4) | 23.6 vs blind 37.9 | 37.7 (2.4) | 40.2 (3.0) | +2.5 [1.6, 3.4] |
| shelf_squatter | tiny | 44.8 (1.7) | 31.1 (2.1) | 38.1 vs blind 31.6 | 31.1 (2.1) | 41.0 (1.9) | +9.9 [9.0, 10.8] |
| shelf_squatter | tiny_choke | 37.0 (2.6) | 28.4 (5.0) | 20.3 vs blind 28.9 | 28.7 (3.7) | 33.5 (4.4) | +5.1 [3.7, 6.5] |
| no_saboteur | tiny | n/a | 64.7 (2.4) | 61.2 vs blind 65.3 | 64.7 (2.4) | 64.7 (2.4) | +0.0 [0.0, 0.0] |
| no_saboteur | tiny_choke | n/a | 55.7 (3.0) | 15.3 vs blind 56.0 | 55.7 (3.0) | 55.7 (3.0) | +0.0 [0.0, 0.0] |
| idle_robot | tiny | n/a | 47.4 (2.6) | 45.9 vs blind 47.7 | 47.4 (2.6) | 47.8 (2.1) | +0.4 [-0.1, 1.0] |
| idle_robot | tiny_choke | n/a | 43.2 (2.2) | 28.6 vs blind 43.5 | 43.2 (2.2) | 43.3 (2.3) | +0.1 [-0.1, 0.3] |

### Throughput retained (V2 seeds, deliveries divided by the no_saboteur fleet on the same layout)

| Behaviour | Layout | Oracle | Blind | V2 threshold | V2 logistic |
|---|---|---|---|---|---|
| goal_blocker | tiny | 0.62 | 0.46 | 0.30 | 0.30 |
| goal_blocker | tiny_choke | 0.47 | 0.39 | 0.17 | 0.11 |
| corridor_parker | tiny | 0.61 | 0.52 | 0.53 | 0.49 |
| corridor_parker | tiny_choke | 0.54 | 0.36 | 0.37 | 0.36 |
| shelf_hider | tiny | 0.73 | 0.63 | 0.63 | 0.68 |
| shelf_hider | tiny_choke | 0.76 | 0.68 | 0.68 | 0.72 |
| shelf_squatter | tiny | 0.69 | 0.48 | 0.48 | 0.63 |
| shelf_squatter | tiny_choke | 0.66 | 0.51 | 0.52 | 0.60 |

### Deadlock rate

| Behaviour | Layout | Oracle | Blind | V1 logistic | V2 threshold | V2 logistic |
|---|---|---|---|---|---|---|
| goal_blocker | tiny | 0.87 | 1.00 | 1.00 | 1.00 | 1.00 |
| goal_blocker | tiny_choke | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| corridor_parker | tiny | 1.00 | 0.97 | 0.97 | 0.97 | 1.00 |
| corridor_parker | tiny_choke | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| shelf_hider | tiny | 0.43 | 0.70 | 0.83 | 0.70 | 0.70 |
| shelf_hider | tiny_choke | 0.23 | 0.60 | 0.87 | 0.60 | 0.37 |
| shelf_squatter | tiny | 0.70 | 0.93 | 0.93 | 0.93 | 0.83 |
| shelf_squatter | tiny_choke | 0.80 | 0.90 | 0.97 | 0.90 | 0.77 |
| no_saboteur | tiny | n/a | 0.07 | 0.17 | 0.07 | 0.07 |
| no_saboteur | tiny_choke | n/a | 0.00 | 1.00 | 0.00 | 0.00 |
| idle_robot | tiny | n/a | 0.47 | 0.57 | 0.47 | 0.47 |
| idle_robot | tiny_choke | n/a | 0.40 | 0.83 | 0.40 | 0.37 |

### Detection (closed loop)

| Behaviour | Layout | Monitor | Detection rate | TTD mean (std) | Honest robot-episodes flagged |
|---|---|---|---|---|---|
| goal_blocker | tiny | V1 logistic | 1.00 | 14.0 (5.9) | 0.97 |
| goal_blocker | tiny | V2 threshold | 1.00 | 140.0 (78.0) | 0.02 |
| goal_blocker | tiny | V2 logistic | 1.00 | 62.7 (28.9) | 0.06 |
| goal_blocker | tiny_choke | V1 logistic | 1.00 | 19.4 (9.7) | 0.74 |
| goal_blocker | tiny_choke | V2 threshold | 0.97 | 158.5 (106.3) | 0.41 |
| goal_blocker | tiny_choke | V2 logistic | 1.00 | 57.2 (19.4) | 0.32 |
| corridor_parker | tiny | V1 logistic | 1.00 | 27.0 (12.8) | 0.46 |
| corridor_parker | tiny | V2 threshold | 0.70 | 320.3 (157.8) | 0.12 |
| corridor_parker | tiny | V2 logistic | 1.00 | 114.6 (59.4) | 0.01 |
| corridor_parker | tiny_choke | V1 logistic | 1.00 | 22.4 (5.6) | 0.81 |
| corridor_parker | tiny_choke | V2 threshold | 0.93 | 199.6 (137.7) | 0.17 |
| corridor_parker | tiny_choke | V2 logistic | 0.97 | 115.0 (109.3) | 0.08 |
| shelf_hider | tiny | V1 logistic | 1.00 | 23.7 (13.1) | 0.41 |
| shelf_hider | tiny | V2 threshold | 0.00 | 491.1 (7.2) | 0.00 |
| shelf_hider | tiny | V2 logistic | 1.00 | 106.1 (54.6) | 0.00 |
| shelf_hider | tiny_choke | V1 logistic | 1.00 | 28.3 (20.9) | 0.64 |
| shelf_hider | tiny_choke | V2 threshold | 0.00 | 490.2 (6.0) | 0.00 |
| shelf_hider | tiny_choke | V2 logistic | 1.00 | 121.9 (73.2) | 0.00 |
| shelf_squatter | tiny | V1 logistic | 1.00 | 29.3 (14.0) | 0.60 |
| shelf_squatter | tiny | V2 threshold | 0.00 | 499.4 (0.6) | 0.00 |
| shelf_squatter | tiny | V2 logistic | 1.00 | 68.4 (27.6) | 0.00 |
| shelf_squatter | tiny_choke | V1 logistic | 1.00 | 30.3 (15.3) | 0.79 |
| shelf_squatter | tiny_choke | V2 threshold | 0.03 | 485.8 (74.6) | 0.00 |
| shelf_squatter | tiny_choke | V2 logistic | 1.00 | 66.8 (34.2) | 0.02 |

### Honest robots flagged when there is no saboteur or only an idle robot (V2 logistic; V1 in brackets)

| Condition | Layout | Robot-episode FPR |
|---|---|---|
| no_saboteur_logistic | tiny | 0.00 (0.48) |
| no_saboteur_logistic | tiny_choke | 0.00 (0.79) |
| idle_robot_logistic | tiny | 0.10 (0.57) |
| idle_robot_logistic | tiny_choke | 0.075 (0.64) |

### What the numbers support

- **Version 2 removes the Version 1 failure.** With no saboteur the logistic fleet equals the blind fleet exactly (64.7 and 55.7 deliveries, no honest robot flagged), and the idle robot case is unchanged within noise.
- **Detection is good, flagging is slower.** Detection rate is 1.00 in 7 of 8 logistic cells (0.97 for `corridor_parker` on `tiny_choke`). TTD is 57 to 122 steps, against 14 to 30 in Version 1, because four consecutive windows are now required and windows are scored every 10 steps.
- **Quarantine helps against `shelf_squatter` and `shelf_hider`.** Paired differences (logistic minus blind) with 95% intervals: `shelf_squatter` +9.9 [9.0, 10.8] on `tiny` and +5.1 [3.7, 6.5] on `tiny_choke`; `shelf_hider` +2.9 [1.6, 4.0] and +2.5 [1.6, 3.4]. No interval contains 0, and logistic beat blind in 77% to 100% of the 30 seeds. For `shelf_squatter` that closes roughly 70% (`tiny`) and 60% (`tiny_choke`) of the gap between the blind and oracle fleets. The oracle fleet is still better everywhere.
- **No measurable effect on `corridor_parker`.** The differences, -2.0 [-6.2, 2.2] and -0.1 [-3.8, 3.4], include 0, as do the idle robot differences (+0.4 and +0.1).
- **`goal_blocker` is worse with either monitor.** Logistic minus blind is -10.5 [-14.3, -6.5] and -16.1 [-17.8, -14.2] deliveries (the threshold rule: -10.2 and -12.3), although the saboteur is detected in every episode; on `tiny_choke` logistic was worse than blind in all 30 seeds. The cause was not tested. Possible reasons, as hypotheses only: the blocker parks near the goals, so a permanent block on its tile or on the nearby aisle can cut honest robots off, and 32% of honest robot-episodes were flagged on `tiny_choke` (6% on `tiny`) while honest robots were queued behind the blocker, a situation the training data may not represent. Deadlock was already 1.00 without a monitor.
- **The threshold rule is not a useful monitor.** It detects only `goal_blocker` and part of `corridor_parker`, never `shelf_hider`, nearly never `shelf_squatter`, and flags 41% of honest robots on `tiny_choke` against `goal_blocker`, which also lowers throughput there.
- **Deadlock rates barely move** except for `shelf_hider` on `tiny_choke` (0.60 to 0.37) and slightly for `shelf_squatter`. Deadlock uses a fixed no-delivery length of 26 (`tiny`) and 30 (`tiny_choke`), so most blocking behaviours deadlock under every fleet.

## Interface changes (update `project.md` section 6, 7 and 9, which is gitignored)

- `FleetContext`: new fields `oracle` (default `True`) and `quarantine`, new method `is_rogue`.
- `build_fleet(..., oracle=True, monitor=None)`; experiment conditions may be `{kind, oracle, monitor}` instead of a plain kind.
- Log schema: new columns `target_x`, `target_y` (nullable) and `flagged`. Old logs still load; `episode_metrics` tolerates a missing `flagged`.
- `Navigator.debug()` also returns `target`.
- Week 6 defaults are unchanged by construction (`is_rogue` reduces to the old test when `oracle=True` and nothing is flagged), but this was not confirmed against the simulator, see below.

## Conventions fixed after review

- **Quarantine is not bypassed.** A quarantined tile stays an obstacle even when it is the robot's target (a target may still be a tile another robot stands on, as in Week 6). Honest robots no longer choose a quarantined goal, return slot or parking tile when another exists; a robot standing on a quarantined tile may leave it. If every goal is quarantined the robot waits rather than entering one.
- **Timing.** In step t the controller reads the flags (`act`), then the monitor sees state t (`update`). That completes the row of step t - 1, so a window ending at `t_end` is scored at state `t_end + 1` and the planners react from step `t_end + 2`. Offline `t_flag` is `t_end`; the closed-loop `flagged` column turns True two steps later. The row of the last step never completes online, so offline windows also stop one step before the episode end.
- **Early detection.** A saboteur flagged before its first sabotage step counts as TTD 0 (floor) and is reported separately as `early_flag_rate`; such a flag is not a false positive, because the robot-level label is the saboteur.
- **Missing targets.** `path_efficiency` is 0 when a robot has no planner target. Only the first step of an episode and random robots lack one, but `train_monitor.py` logs the share of rows without a target for saboteurs and honest robots, so a gap between the two would show up before the model is trusted.
- **Cross-validation.** `fit_logistic` raises if a fold lacks either class, instead of returning NaN AUC. Folds are grouped by `episode_seed`.

## Tests

`python -m pytest -q`: 74 passed, 1 warning (a gymnasium reward-type warning that predates Week 7) in 4.8 s on the final run. This includes the Week 6 tests that need the simulator, and `test_online_monitor_rows_match_the_episode_logger`, which confirms that the delivered, picked_up, dropped and moved events the live monitor infers match the simulator's own log. Version 2 added `tests/test_monitor_selection.py` (7 tests: persistence needs consecutive windows in time order, the replay matches `first_flag`, robot-episode FPR counts robots not windows, a silent monitor is not chosen, nothing-useful and empty-feasible-set cases are reported). The empty-feasible-set test was added after the first Version 2 training run crashed on exactly that case; the earlier tests missed it because their synthetic candidates always included a never-flagging threshold, which the real candidate grid did not.

An earlier full run (67 tests) took 37.6 s, more than the 30 s convention for the Week 6 suite; the final run took 4.8 s, so the time was not investigated further.

Week 6 regression: `run_baseline.py` on the Week 6 config reproduces the committed `week6_baseline.csv` on all 16 compared columns.

## Commands that were run

```bash
python -m pytest -q
python scripts/run_baseline.py --config configs/week7_data.yaml                       # 720 episodes
python scripts/train_monitor.py                                                       # Version 1
python scripts/eval_monitor.py                                                        # Version 1 closed loop, seeds 3000 to 3029
python scripts/train_monitor.py --selection episode --suffix _v2                      # Version 2
python scripts/eval_monitor.py --config configs/week7_eval_v2.yaml                    # Version 2 closed loop, seeds 3100 to 3129
python scripts/paired_analysis.py                                                     # paired differences, writes week7_eval_v2_paired.csv
python scripts/make_gif.py --config configs/week7_eval_v2.yaml --conditions shelf_squatter_blind shelf_squatter_logistic goal_blocker_blind goal_blocker_logistic --layout tiny_choke
```

## GIFs

`scripts/make_gif.py --config configs/week7_eval_v2.yaml --conditions shelf_squatter_blind shelf_squatter_logistic goal_blocker_blind goal_blocker_logistic --layout tiny_choke` wrote four files to `docs/media/`: `shelf_squatter_blind_tiny_choke.gif`, `shelf_squatter_logistic_tiny_choke.gif`, `goal_blocker_blind_tiny_choke.gif` and `goal_blocker_logistic_tiny_choke.gif`. Each is one real episode, seed 3100 (the first seed of the config, fixed before recording, not chosen from the results), `tiny_choke`, first 300 steps, every second frame. They are single episodes, not typical behaviour: the averages and intervals over 30 seeds are in the tables above, and `goal_blocker` is a case where the logistic monitor lowers throughput.

The saboteur is outlined in red and a robot the monitor has flagged gets a yellow outline outside it. In the two logistic GIFs the yellow outline first appears at about step 52 (`shelf_squatter`) and step 48 (`goal_blocker`); the blind GIFs have none. The `goal_blocker` GIFs hold fewer frames (138 blind, 105 logistic) than the 150 of the other two. This is not an early end of the episode: the frames of each GIF sum to the same total playing time (about 18 s), because the GIF writer merges identical consecutive frames and the `goal_blocker` scenes stand still for long stretches. The evaluation itself is unaffected: all 44 condition and layout runs (30 episodes each) ran the full 500 steps.

## Not yet done

- Intervals describe these 30 seeds per cell only. There is no correction for the many comparisons in the tables, and the bootstrap treats the 30 seeds as independent draws.

## Limitations

- One training run and one closed-loop run per version; no repeated runs. Paired bootstrap intervals cover Version 2 only.
- Version 2 was designed after seeing Version 1 closed-loop results. Seeds 3100 to 3129 were not used for any choice, but the decision to change the selection rule was informed by Version 1.
- Quarantine is permanent and blocks the flagged tile as it was at flag time, as in the proposal. Version 2 does not change the quarantine semantics or the metric definitions. A time-limited or probabilistic quarantine is untested.
- `goal_blocker` remains a negative result; the reason is not established.
- Version 2 detects slowly (TTD 57 to 122 steps for the logistic monitor). TTD has a granularity of 10 steps, and no flag can occur before step 20, plus the persistence requirement.
- The logistic coefficients are dominated by `unrequested_carried` (-35.3) and `deliveries` (-20.5). They show what the model relies on, not why; with correlated features the signs should not be read causally. Feature 9 and the planned tiles rely on the robots' own reports.
- The blind fleet in the data config differs from the Week 6 oracle fleet, so Week 7 data are not comparable step by step with Week 6 logs.
