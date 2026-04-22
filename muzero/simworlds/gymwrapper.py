"""Gymnasium environment adapter for any discrete-action env.

MuZero's u-MCTS uses the LEARNED model (NN_d) for lookahead, so during a tree
search we never need to step the real env. This means we can get away with a
stateful wrapper: the gym env lives inside the SimWorld, and GameState is just
a cached observation + terminal flag.

Why this is safe:
  - RLM always calls simworld.step(state, action) on the latest state (never
    rewinds), so the real env's internal state stays in lock-step with the
    'state' object being passed around.
  - MCTS never touches simworld.step — it expands via asm.child, which calls
    NN_d. Zero risk of the real env being stepped mid-simulation.

Supported games need: Box-shaped continuous OR discrete observation, and
Discrete action space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import SimWorld


@dataclass
class GymState:
    obs: np.ndarray
    terminal: bool

    def observable(self) -> np.ndarray:
        return self.obs.astype(np.float32)


class GymSimWorld(SimWorld):
    """Wrap any Gymnasium env with a discrete action space."""

    def __init__(self, env_id: str, max_episode_steps: int | None = None,
                 reward_scale: float = 1.0, **env_kwargs: Any):
        import gymnasium as gym

        self._env_id = env_id
        self._env_kwargs = dict(env_kwargs)
        if max_episode_steps is not None:
            self._env_kwargs["max_episode_steps"] = max_episode_steps
        self._reward_scale = float(reward_scale)
        self._env = gym.make(env_id, **self._env_kwargs)

        obs_space = self._env.observation_space
        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            self._state_shape = tuple(int(d) for d in obs_space.shape)
        else:
            raise ValueError(f"Unsupported observation space: {obs_space}")

        act_space = self._env.action_space
        if not hasattr(act_space, "n"):
            raise ValueError(
                f"GymSimWorld requires Discrete action space, got {act_space}"
            )
        self._num_actions = int(act_space.n)

        # Renderer (built lazily; separate env with render_mode set).
        self._render_env = None

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self._state_shape

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def num_players(self) -> int:
        return 1

    @property
    def reward_range(self) -> tuple[float, float]:
        lo, hi = self._env.reward_range
        return (float(lo) * self._reward_scale, float(hi) * self._reward_scale)

    def initial_state(self, rng: np.random.Generator) -> GymState:
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = self._env.reset(seed=seed)
        return GymState(obs=np.asarray(obs, dtype=np.float32), terminal=False)

    def step(self, state: GymState, action: int) -> tuple[GymState, float, bool]:
        obs, reward, terminated, truncated, _ = self._env.step(int(action))
        terminal = bool(terminated or truncated)
        scaled_r = float(reward) * self._reward_scale
        return (GymState(obs=np.asarray(obs, dtype=np.float32), terminal=terminal),
                scaled_r, terminal)

    def legal_actions(self, state: GymState) -> np.ndarray:
        return np.ones(self._num_actions, dtype=bool)

    def current_player(self, state: GymState) -> int:
        return 0

    def is_terminal(self, state: GymState) -> bool:
        return bool(state.terminal)

    def blank_state(self) -> GymState:
        return GymState(obs=np.zeros(self._state_shape, dtype=np.float32),
                        terminal=False)

    def render_frame(self, surface: Any, state: GymState) -> None:
        # Render via a separate env instance that's configured for rgb_array
        # output. We step it in lock-step with the main env if rendering is on.
        import gymnasium as gym
        import pygame

        if self._render_env is None:
            self._render_env = gym.make(self._env_id, render_mode="rgb_array",
                                        **{k: v for k, v in self._env_kwargs.items()
                                           if k != "render_mode"})
            self._render_env.reset()
        try:
            frame = self._env.render() if getattr(self._env, "render_mode", None) == "rgb_array" else None
            if frame is None:
                # Main env wasn't built for rendering — just fill grey.
                surface.fill((40, 40, 48))
                font = pygame.font.Font(None, 20)
                txt = font.render(f"{self._env_id} (render disabled)", True, (200, 200, 200))
                surface.blit(txt, (10, 10))
                return
            import numpy as np

            arr = np.asarray(frame)
            arr = np.transpose(arr, (1, 0, 2))
            frame_surf = pygame.surfarray.make_surface(arr)
            sw, sh = surface.get_size()
            frame_surf = pygame.transform.scale(frame_surf, (sw, sh))
            surface.blit(frame_surf, (0, 0))
        except Exception:
            surface.fill((40, 40, 48))
