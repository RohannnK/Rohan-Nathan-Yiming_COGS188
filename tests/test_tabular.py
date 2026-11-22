"""Week 8 tabular experiment: gridworld model, value iteration, SARSA vs. Q-learning."""
import numpy as np
import pytest

from warehouse_sabotage.rl.dp import policy_evaluation, value_iteration
from warehouse_sabotage.rl.gridworld import AisleGridworld
from warehouse_sabotage.rl.tabular import epsilon, train


def test_transition_model_is_a_distribution():
    env = AisleGridworld(bump_prob=0.05)
    assert np.allclose(env.prob.sum(axis=2), 1.0)


def test_value_iteration_finds_the_aisle_without_bumps():
    env = AisleGridworld(bump_prob=0.0)
    v, pi = value_iteration(env, gamma=1.0)
    assert env.route(pi) == "aisle"
    assert len(env.greedy_path(pi)) - 1 == 11
    assert v[env.start] == pytest.approx(-11.0)


def test_value_iteration_prefers_the_detour_when_bumps_are_likely():
    env = AisleGridworld(bump_prob=0.05)
    v, pi = value_iteration(env, gamma=1.0)
    assert env.route(pi) == "detour"
    assert v[env.start] == pytest.approx(-15.0)


def test_policy_evaluation_of_the_optimal_policy_matches_v_star():
    env = AisleGridworld(bump_prob=0.01)
    v, pi = value_iteration(env, gamma=0.99)
    assert np.allclose(policy_evaluation(env, pi, 0.99)[env.start], v[env.start])


def test_linear_epsilon_schedule():
    sched = {"kind": "linear", "start": 1.0, "end": 0.1, "frac": 0.5}
    assert epsilon(sched, 0, 100) == 1.0
    assert epsilon(sched, 25, 100) == pytest.approx(0.55)
    assert epsilon(sched, 90, 100) == pytest.approx(0.1)


def test_cliff_walking_result_q_learning_aisle_sarsa_detour():
    """With constant epsilon, Q-learning learns the optimal aisle and SARSA the safer detour (Sutton and Barto)."""
    env = AisleGridworld(bump_prob=0.0)
    sched = {"kind": "constant", "eps": 0.1}
    q_q, _ = train(env, "q_learning", 300, 0.5, 1.0, sched, np.random.default_rng(0))
    q_s, _ = train(env, "sarsa", 300, 0.5, 1.0, sched, np.random.default_rng(0))
    assert env.route(q_q.argmax(axis=1)) == "aisle"
    assert env.route(q_s.argmax(axis=1)) == "detour"
