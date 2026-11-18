import numpy as np
import pandas as pd

from warehouse_sabotage.eval.metrics import paired_difference


def frame(treat, control, layout="tiny"):
    seeds = range(len(treat))
    return pd.concat([pd.DataFrame({"condition": c, "layout": layout, "episode_seed": list(seeds), "deliveries": v})
                      for c, v in (("logistic", treat), ("blind", control))], ignore_index=True)


def test_paired_difference_uses_the_same_seed_in_both_conditions():
    # unpaired the means differ by 2.0, paired every episode differs by exactly 2: the interval collapses to 2
    treat, control = [10, 20, 30, 40], [8, 18, 28, 38]
    r = paired_difference(frame(treat, control), "logistic", "blind").iloc[0]
    assert r["episodes"] == 4 and r["mean_diff"] == 2 and r["ci_low"] == 2 and r["ci_high"] == 2
    assert r["share_improved"] == 1


def test_paired_difference_interval_covers_the_mean_and_is_seeded():
    rng = np.random.default_rng(1)
    control = rng.integers(20, 40, 30)
    treat = control + rng.integers(-3, 8, 30)
    f = frame(list(treat), list(control))
    a, b = paired_difference(f, "logistic", "blind"), paired_difference(f, "logistic", "blind")
    assert a.equals(b) and a["ci_low"].iat[0] <= a["mean_diff"].iat[0] <= a["ci_high"].iat[0]
    assert a["mean_diff"].iat[0] == (treat - control).mean()


def test_paired_difference_drops_seeds_missing_from_one_condition_and_splits_layouts():
    f = pd.concat([frame([5, 6, 7], [4, 4, 4], "tiny"), frame([1, 1], [3, 3], "tiny_choke")], ignore_index=True)
    f = f[~((f["condition"] == "blind") & (f["layout"] == "tiny") & (f["episode_seed"] == 2))]
    r = paired_difference(f, "logistic", "blind").set_index("layout")
    assert r.loc["tiny", "episodes"] == 2 and r.loc["tiny", "mean_diff"] == 1.5
    assert r.loc["tiny_choke", "mean_diff"] == -2 and r.loc["tiny_choke", "share_improved"] == 0
