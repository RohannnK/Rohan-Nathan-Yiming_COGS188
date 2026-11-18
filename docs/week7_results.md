# Week 7: runtime monitor and quarantine-and-replan

Status: all seven Week 7 deliverables are implemented on branch `feat/week7-monitor`. **No experimental results are reported here.** The environment this work was written in had no rware, gymnasium or pyarrow and no network access, so no episodes could be generated, and the Week 6 tests that need the simulator could not be run. Everything below the "Not yet run" heading must be executed on a machine with the `cogs188` environment before any number is quoted. The tables of results are intentionally left out.

## What was implemented

| Piece | File | What it does |
|---|---|---|
| Features | `monitor/features.py` | The 12 proposal features over windows of 20 steps, stride 10 (definitions in the module docstring) |
| Dataset | `monitor/dataset.py` | Window rows from episode logs, robot-level label `is_saboteur`, window-level label `sabotage_window`, split by episode seed |
| Threshold monitor | `monitor/threshold.py` | Flag a window when `choke_steps` or `idle_ticks` reaches a limit; limits chosen on the validation split |
| Logistic regression | `monitor/logistic.py` | StandardScaler + balanced `LogisticRegression`, C by `GroupKFold` on the training split, tau chosen on the validation split |
| Quarantine | `monitor/quarantine.py` | `QuarantineMonitor` (Monitor protocol): scores the live episode with the same feature code, flags permanently, `save_monitor` / `load_monitor` |
| Metrics | `eval/metrics.py` | Window precision, recall, F1, ROC-AUC; TTD, detection rate, robot-episode FPR (offline and closed loop) |
| Scripts, configs | `scripts/train_monitor.py`, `scripts/eval_monitor.py`, `configs/week7_data.yaml`, `configs/week7_eval.yaml` | Data generation reuses `run_baseline.py` |

## Decisions worth explaining

1. **Oracle versus blind fleet.** In Week 6 `build_fleet` excludes the saboteur from `honest_ids`, so the honest planners treated it as an obstacle from step 0, which means a monitor could not add anything. `FleetContext.oracle` (default `True`, Week 6 behaviour unchanged) now says whether the planners know. With `oracle=False` the saboteur is treated as a cooperating robot until the monitor flags it. The evaluation compares three fleets per behaviour: `_oracle` (Week 6, an upper bound), `_blind` (no monitor, the lower bound) and `_threshold` / `_logistic` (blind until flagged).
2. **Quarantine.** When a robot is first flagged, the tile it stands on is blocked for the honest planners for the rest of the episode (`FleetContext.quarantine`), as in the proposal. The flagged robot's current tile is blocked as well, through the existing `ctx.flagged` hook. The test `test_detection_changes_the_honest_route` checks that the planned route changes once a robot is flagged.
3. **Same code online and offline.** `QuarantineMonitor` builds log-schema rows from `WarehouseState` and the actions, and calls the same `window_table` as the dataset builder. `test_online_features_match_the_logged_features` compares both on a scripted episode and requires identical features.
4. **Leakage.** Features never read `is_saboteur`, `sabotage_active`, `task_phase` or planner diagnostics (`test_features_ignore_label_columns`). Seeds are shared by every condition and layout, so episodes with the same seed have the same spawns, requests and saboteur id; the split groups by `episode_seed`, not by run. The scaler is fitted on the training windows only. Thresholds and tau are chosen on the validation split, the test split is used once for reporting. Closed-loop evaluation uses seeds 3000 to 3029, which never touch training.
5. **Two assumptions to state in the report.** Robots report their planned tiles and current target to the fleet manager (the monitor reads them from the controller's `debug_info`, the same values the logs hold). Feature 7 counts any other robot that stopped on a highway tile for 3 steps, since the monitor cannot know who is honest.
6. **Feature definitions the proposal leaves open.** Corridor tiles are chokepoints that are not goal or goal access tiles, so `corridor_frac` and `choke_steps` differ; `requested_undelivered` is requested pick-ups minus deliveries in the window (at least 0); `path_efficiency` is the shortest-path distance covered towards the target held at the window start, per step (0 without a target).

## Interface changes (update `project.md` section 6, 7 and 9, which is gitignored)

- `FleetContext`: new fields `oracle` (default `True`) and `quarantine`, new method `is_rogue`.
- `build_fleet(..., oracle=True, monitor=None)`; experiment conditions may be `{kind, oracle, monitor}` instead of a plain kind.
- Log schema: new columns `target_x`, `target_y` (nullable) and `flagged`. Old logs still load; `episode_metrics` tolerates a missing `flagged`.
- `Navigator.debug()` also returns `target`.
- Week 6 defaults are unchanged by construction (`is_rogue` reduces to the old test when `oracle=True` and nothing is flagged), but this was not confirmed against the simulator, see below.

## Tests

Run in the authoring environment (39 passed in about 3 s): `tests/test_planning.py` (the 12 Week 6 planning tests), `tests/test_monitor_features.py` (11), `tests/test_monitor_models.py` (10), `tests/test_quarantine.py` (6). They use hand-made logs and states, no simulator.

Written but not run: `tests/test_rollout.py` (two new tests; the Week 6 tests in that file, `test_env_*` and `test_layouts` also need rware and gymnasium).

## Not yet run (needs the `cogs188` environment)

```bash
pytest                                                              # all Week 6 and Week 7 tests; Week 6 numbers must be unchanged
python scripts/run_baseline.py --smoke                              # Week 6 smoke test
python scripts/run_baseline.py --config configs/week7_data.yaml     # 720 episodes, about 4 minutes
python scripts/train_monitor.py                                     # fits both monitors, writes results/summary/week7_monitor.csv
python scripts/eval_monitor.py                                      # closed loop, 22 conditions x 2 layouts x 30 seeds
python scripts/make_gif.py --config configs/week7_eval.yaml --conditions goal_blocker_blind goal_blocker_logistic --layout tiny_choke   # reroute GIF, needs a display
```

## Limitations

- `shelf_hider` is expected to stay hard to detect (Week 6 finding); report it as it comes out.
- Feature 9 and the planned tiles use the robots' own reports; a monitor that must work without them would need a different target estimate.
- Windows score every 10 steps after a 20 step warm-up, so TTD has a granularity of 10 steps and no flag can occur before step 20.
- The blind fleet in the data config differs from the Week 6 oracle fleet, so Week 7 data are not comparable step by step with Week 6 logs.
