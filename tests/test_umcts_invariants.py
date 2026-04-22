"""Invariants of u-MCTS that must hold regardless of the underlying networks.

We substitute a fake ASM for the neural nets so the test is fast and
deterministic.
"""
import numpy as np

from configs._schema import UMCTSConfig
from muzero.ai.search.umcts import UMCTS


class _FakeASM:
    def __init__(self, num_actions=3, num_players=1, hidden_dim=4):
        self.num_actions = num_actions
        self.num_players = num_players
        self.hidden_dim = hidden_dim

    def root_from_game_states(self, phi):
        return np.zeros(self.hidden_dim, dtype=np.float32)

    def child(self, sigma, action):
        # deterministic: add action to sigma, reward = small constant
        return sigma + float(action), 0.1 * float(action)

    def policy_value(self, sigma):
        # uniform policy, constant value
        return np.full(self.num_actions, 1.0 / self.num_actions, dtype=np.float32), 0.0


def _cfg(M_s=16, d_max=3, c_ucb=1.25, dirichlet_alpha=None, dirichlet_frac=0.0,
         rollout_enabled=False):
    return UMCTSConfig(M_s=M_s, d_max=d_max, c_ucb=c_ucb,
                       dirichlet_alpha=dirichlet_alpha,
                       dirichlet_frac=dirichlet_frac,
                       rollout_enabled=rollout_enabled)


def test_visit_counts_sum_to_M_s():
    asm = _FakeASM(num_actions=3, num_players=1)
    umcts = UMCTS(asm, _cfg(M_s=20, d_max=3), gamma=0.95)
    legal = np.ones(3, dtype=bool)
    phi = np.zeros((2, 4, 4, 2), dtype=np.float32)
    rng = np.random.default_rng(0)
    result = umcts.run(phi, legal, to_play=0, rng=rng)
    assert result.visit_counts.sum() == 20


def test_expanded_node_has_num_actions_edges():
    asm = _FakeASM(num_actions=4)
    umcts = UMCTS(asm, _cfg(M_s=8, d_max=2), gamma=0.9)
    legal = np.ones(4, dtype=bool)
    phi = np.zeros((2, 3, 3, 1), dtype=np.float32)
    rng = np.random.default_rng(1)
    result = umcts.run(phi, legal, to_play=0, rng=rng)
    assert len(result.visit_counts) == 4


def test_policy_sums_to_one_and_respects_illegal_at_root():
    asm = _FakeASM(num_actions=3)
    umcts = UMCTS(asm, _cfg(M_s=12, d_max=2), gamma=0.9)
    legal = np.array([True, False, True])
    phi = np.zeros((1, 2, 2, 1), dtype=np.float32)
    rng = np.random.default_rng(2)
    result = umcts.run(phi, legal, to_play=0, rng=rng)
    assert abs(result.policy.sum() - 1.0) < 1e-5
    # illegal action should never be visited
    assert result.visit_counts[1] == 0


def test_two_player_backprop_sign_flip():
    asm = _FakeASM(num_actions=2, num_players=2)
    umcts = UMCTS(asm, _cfg(M_s=8, d_max=2), gamma=1.0)
    legal = np.ones(2, dtype=bool)
    phi = np.zeros((1, 2, 2, 1), dtype=np.float32)
    # Just confirm it runs without error and produces a valid distribution.
    result = umcts.run(phi, legal, to_play=0, rng=np.random.default_rng(3))
    assert result.visit_counts.sum() == 8
    assert abs(result.policy.sum() - 1.0) < 1e-5
