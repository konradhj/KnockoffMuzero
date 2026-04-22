"""EpisodeBuffer: stores full episodes as numpy arrays, samples minibatches
shaped exactly as the unrolled BPTT loss expects.

Slicing convention:
  At random step index k in a chosen episode:
    phi_stack[b] = states[k-q:k+1]          # (q+1, *state_shape)   left-padded with blank
    actions[b, j] = actions[k+j]            j = 0..w-1
    target_pi[b, j] = policies[k+j]         j = 0..w
    target_v[b, j] = n-step return bootstrapped from root_values[k+j+n]
    target_r[b, j] = rewards[k+j]           j = 0..w-1
    mask[b, j] = 1 if step (k+j) is within the episode and non-terminal-overshoot else 0
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from configs._schema import TrainingConfig
from muzero.ai.types import MinibatchArrays


@dataclass
class EpisodeRecord:
    states: np.ndarray       # (T+1, *state_shape) float32
    actions: np.ndarray      # (T,) int32
    rewards: np.ndarray      # (T,) float32
    policies: np.ndarray     # (T, num_actions) float32
    root_values: np.ndarray  # (T,) float32
    terminal: bool


@dataclass
class EpisodeBuilder:
    """Incrementally assembles an EpisodeRecord during an episode."""
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    root_values: list[float] = field(default_factory=list)

    def append_step(self, state_obs: np.ndarray, action: int, reward: float,
                    policy: np.ndarray, root_value: float) -> None:
        self.states.append(state_obs.astype(np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.policies.append(policy.astype(np.float32))
        self.root_values.append(float(root_value))

    def append_final_state(self, state_obs: np.ndarray) -> None:
        self.states.append(state_obs.astype(np.float32))

    def build(self, terminal: bool) -> EpisodeRecord:
        return EpisodeRecord(
            states=np.stack(self.states, axis=0) if self.states else np.zeros((0,)),
            actions=np.asarray(self.actions, dtype=np.int32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            policies=np.stack(self.policies, axis=0) if self.policies else np.zeros((0,)),
            root_values=np.asarray(self.root_values, dtype=np.float32),
            terminal=terminal,
        )


class EpisodeBuffer:
    def __init__(self, cfg: TrainingConfig, state_shape: tuple[int, ...],
                 num_actions: int, blank_obs: np.ndarray):
        self._cfg = cfg
        self._state_shape = tuple(state_shape)
        self._num_actions = int(num_actions)
        self._blank_obs = blank_obs.astype(np.float32)
        assert self._blank_obs.shape == self._state_shape, (
            f"blank_obs shape {self._blank_obs.shape} != state_shape {self._state_shape}"
        )
        self._episodes: deque[EpisodeRecord] = deque(maxlen=cfg.buffer_capacity)

    def append(self, ep: EpisodeRecord) -> None:
        if len(ep.actions) == 0:
            return
        self._episodes.append(ep)

    def __len__(self) -> int:
        return len(self._episodes)

    def _uniform_policy(self) -> np.ndarray:
        return np.full(self._num_actions, 1.0 / self._num_actions, dtype=np.float32)

    def _value_target(self, ep: EpisodeRecord, k_j: int) -> float:
        """n-step bootstrap target at step k_j within ep.

        target_v = sum_{i=0..n-1} gamma^i * rewards[k_j + i]  +  gamma^n * root_values[k_j + n]
        with n = min(n_step, T - k_j). If k_j + n reaches T and episode is terminal,
        we drop the bootstrap term (no future value).
        """
        T = int(len(ep.actions))
        if k_j >= T:
            return 0.0
        n_step = self._cfg.n_step
        gamma = self._cfg.gamma
        n = min(n_step, T - k_j)
        g = 0.0
        for i in range(n):
            g += (gamma ** i) * float(ep.rewards[k_j + i])
        bootstrap_idx = k_j + n
        if bootstrap_idx < T:
            g += (gamma ** n) * float(ep.root_values[bootstrap_idx])
        elif not ep.terminal:
            # episode truncated but not terminal: bootstrap from final stored value
            g += (gamma ** n) * float(ep.root_values[T - 1])
        return g

    def sample_minibatch(self, rng: np.random.Generator) -> MinibatchArrays:
        mbs = self._cfg.mbs
        q = self._cfg.q
        w = self._cfg.w
        A = self._num_actions

        phi_stack = np.zeros((mbs, q + 1, *self._state_shape), dtype=np.float32)
        actions = np.zeros((mbs, w), dtype=np.int32)
        target_pi = np.zeros((mbs, w + 1, A), dtype=np.float32)
        target_v = np.zeros((mbs, w + 1), dtype=np.float32)
        target_r = np.zeros((mbs, w), dtype=np.float32)
        mask = np.zeros((mbs, w + 1), dtype=np.float32)

        eps = list(self._episodes)
        for b in range(mbs):
            ep = eps[rng.integers(0, len(eps))]
            T = int(len(ep.actions))
            if T == 0:
                target_pi[b, :] = self._uniform_policy()
                continue
            k = int(rng.integers(0, T))

            # phi_stack: states[k-q:k+1] with left-padding
            for i in range(q + 1):
                src = k - q + i
                if src < 0:
                    phi_stack[b, i] = self._blank_obs
                else:
                    phi_stack[b, i] = ep.states[src]

            # actions and target_r for unroll j = 0..w-1
            for j in range(w):
                src = k + j
                if src < T:
                    actions[b, j] = ep.actions[src]
                    target_r[b, j] = ep.rewards[src]
                else:
                    actions[b, j] = 0
                    target_r[b, j] = 0.0

            # target_pi and target_v, mask
            for j in range(w + 1):
                src = k + j
                if src < T:
                    target_pi[b, j] = ep.policies[src]
                    target_v[b, j] = self._value_target(ep, src)
                    mask[b, j] = 1.0
                else:
                    target_pi[b, j] = self._uniform_policy()
                    target_v[b, j] = 0.0
                    mask[b, j] = 0.0

        return MinibatchArrays(phi_stack=phi_stack, actions=actions,
                               target_pi=target_pi, target_v=target_v,
                               target_r=target_r, mask=mask)
