"""Week 8 DQN: fleet wrapper, shaping, Double DQN target and a short training run."""
import numpy as np
import torch

from warehouse_sabotage.rl.dqn import DQNAgent, DuelingQNet, linear_eps, train_dqn
from warehouse_sabotage.rl.fleet_env import FleetEnv

CFG = {"layout": "tiny", "env": {"max_steps": 60}, "kind": "honest", "eval_seeds": [5],
       "dqn": {"double": True, "hidden": 16, "lr": 1e-3, "gamma": 0.9, "batch_size": 8, "buffer_size": 500,
               "learning_starts": 16, "train_every": 1, "target_update": 20, "eps_start": 1.0, "eps_end": 0.1,
               "eps_decay_frac": 0.5, "pickup_bonus": 1.0, "total_steps": 120, "smoke_steps": 120,
               "eval_every": 1, "eval_eps": 0.0}}


def test_learner_is_never_the_saboteur_and_obs_is_local():
    env = FleetEnv("tiny", {"max_steps": 20}, kind="goal_blocker")
    for seed in range(5):
        obs = env.reset(seed)
        assert env.rid != env.sab
        assert obs.shape == (71,)
        assert env.rid not in env.fleet.ctx.honest_ids  # A* robots treat the learner as an obstacle


def test_astar_slot_matches_the_honest_fleet():
    env = FleetEnv("tiny", {"max_steps": 200}, learner="astar")
    env.reset(0)
    for _ in range(200):
        env.step(0)
    assert env.stats["learner_deliveries"] > 0


def test_paid_bonuses_are_forgotten_once_a_shelf_is_no_longer_requested():
    env = FleetEnv("tiny", {"max_steps": 500}, pickup_bonus=1.0)
    env.reset(0)
    env.bonus_paid = {env.state.requested_shelf_ids[0]}
    env.step(0)
    assert env.bonus_paid <= set(env.state.requested_shelf_ids)


def test_dueling_head_is_mean_zero_advantage():
    net = DuelingQNet(4, 3, 8)
    x = torch.randn(5, 4)
    h = net.body(x)
    assert torch.allclose(net(x).mean(dim=1), net.value(h).squeeze(1), atol=1e-6)


def test_double_target_uses_online_argmax():
    agent = DQNAgent(2, 2, {**CFG["dqn"], "gamma": 1.0, "batch_size": 1}, np.random.default_rng(0))
    for net, w in ((agent.online, [[0.0, 1.0]]), (agent.target, [[5.0, 2.0]])):
        for p in net.parameters():
            torch.nn.init.zeros_(p)
        net.adv.bias.data = torch.tensor(w[0])
    agent.buffer.add(np.zeros(2), 0, 0.0, np.zeros(2), 0.0)
    s, a, r, s2, d = agent.buffer.sample(1, agent.rng)
    a2 = agent.online(s2).argmax(dim=1, keepdim=True)
    assert a2.item() == 1 and agent.target(s2).gather(1, a2).item() == -1.5  # 2 - mean(5, 2), not max = 1.5


def test_linear_eps():
    assert linear_eps(0, 1.0, 0.1, 10) == 1.0 and abs(linear_eps(20, 1.0, 0.1, 10) - 0.1) < 1e-12


def test_short_training_run_logs_episodes_and_evaluations():
    curves = train_dqn(CFG, seed=1, smoke=True)
    assert curves["step"].iloc[-1] == 120 and len(curves) == 2
    assert curves["eval_learner_deliveries"].notna().all()
