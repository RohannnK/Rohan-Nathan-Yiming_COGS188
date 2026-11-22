# Week 8 (part C): moving to the small map

```bash
python scripts/run_baseline.py --config configs/week8_small_baseline.yaml   # about 80 s
```

Output: `results/summary/week8_small_baseline.csv`. Same fleet, saboteur settings, conditions and seeds (1000 to 1029) as the Week 6 baseline; only the layout changes.

## Layout choice

We use rware's stock small map unchanged (`LAYOUTS["small"]`, 20 x 10, goals at (4, 19) and (5, 19)). A test checks it tile for tile against `rware-small-4ag-v2`. A separate choke variant was not needed: the top shelf block already has four 1-wide vertical corridors (x = 0, 3, 6, 9, rows 1 to 8), like `tiny_choke`, and the bottom block is open, like `tiny`. Removing any single chokepoint keeps the highway graph connected (tested, as for the tiny layouts). Delivery trips are longer because 48 of the 80 shelf slots (60%) sit 11 or more rows from the goals.

This settles the open question in project.md section 13 ("stock small vs. a choke variant").

## Deadlock threshold

Same rule as Week 6: the rounded 95th percentile of the longest no-delivery gap per `no_saboteur` episode. On small that is 35.55, so **L = 36** (tiny 26, tiny_choke 30). It was set after a first run and should stay fixed from now on.

## Results (30 episodes per condition, mean ± std)

| condition | deliveries | ratio to no saboteur | deadlock rate | longest gap (steps) | tiny ratio (Week 6) |
|---|---|---|---|---|---|
| random_policy | 0.0 ± 0.2 | 0.00 | 1.00 | 492.9 ± 38.7 | 0.00 |
| no_saboteur | 45.3 ± 2.2 | 1.00 | 0.07 | 29.3 ± 4.4 | 1.00 |
| idle_robot | 34.5 ± 1.6 | 0.76 | 0.47 | 34.2 ± 5.5 | 0.77 |
| goal_blocker | 30.1 ± 1.4 | 0.67 | 0.80 | 42.2 ± 7.1 | 0.63 |
| corridor_parker | 26.7 ± 2.5 | 0.59 | 0.90 | 48.4 ± 9.6 | 0.63 |
| shelf_hider | 33.6 ± 1.6 | 0.74 | 0.47 | 35.4 ± 4.4 | 0.75 |
| shelf_squatter | 31.1 ± 1.7 | 0.69 | 0.70 | 38.0 ± 6.7 | 0.70 |

Paired difference to the idle control (same seeds, bootstrap 95% interval, `eval.metrics.paired_difference`):

| behaviour | saboteur minus idle | 95% interval | seeds where the saboteur did better |
|---|---|---|---|
| goal_blocker | -4.4 | [-4.9, -3.9] | 0% |
| corridor_parker | -7.9 | [-8.7, -7.0] | 0% |
| shelf_squatter | -3.4 | [-4.0, -2.8] | 0% |
| shelf_hider | -1.0 | [-1.5, -0.5] | 13% |

## Findings

- **The A* fleet still works on the larger map.** No-saboteur throughput is 45.3 deliveries (against 0.03 for random robots), with a low deadlock rate (0.07, by construction of L). Throughput is lower than on tiny (64.7) because trips are longer.
- **Every behaviour costs more than losing a worker**, and the Week 6 acceptance criterion holds on small: all four intervals against the idle control exclude 0.
- **`corridor_parker` becomes the most harmful behaviour** (ratio 0.59, -7.9 against idle). On small it has real 1-wide corridors far from the goals to park in. On tiny it tied with `goal_blocker`.
- **`goal_blocker` is relatively weaker on small** (0.67 against 0.63 on tiny), as more of the work is far from the goals.
- **`shelf_hider` retest (promised in section 13):** it is still the weakest behaviour (-1.0 against idle, interval [-1.5, -0.5]). Hidden shelves now land farther away, but the cost is still close to losing one worker. We keep it as a documented low-harm behaviour rather than tuning it.
- **Deadlock rates rise for every saboteur condition** (0.47 to 0.90). Gaps are longer on a bigger map, so this mostly reflects the threshold, and the longest-gap column is the better comparison.
- Planning cost per environment step is similar to tiny (0.05 to 0.15 ms; 1.1 ms for `corridor_parker`, which triggers many replans).

## Not done here

- The Week 7 monitor was trained on tiny episodes. Its transfer to small is a Week 9 generalisation test, so it was not run here.
- The DQN learner (Week 8 part B) is trained on tiny. Moving it to small is one config change (`layout: small`), but small is harder to explore, so it waits until it learns on tiny.
