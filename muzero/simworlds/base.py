from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np


class GameState(Protocol):
    """Opaque to the AI core. Only the observable() method is required by NN input."""

    def observable(self) -> np.ndarray:
        """Return a float32 array of shape SimWorld.state_shape."""
        ...


class SimWorld(ABC):
    """The critical divide lives here: everything game-specific stops at this interface."""

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """Shape of observable(). For grid games: (rows, cols, channels)."""

    @property
    @abstractmethod
    def num_actions(self) -> int:
        ...

    @property
    @abstractmethod
    def num_players(self) -> int:
        ...

    @property
    @abstractmethod
    def reward_range(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def initial_state(self, rng: np.random.Generator) -> GameState:
        ...

    @abstractmethod
    def step(self, state: GameState, action: int) -> tuple[GameState, float, bool]:
        """Returns (next_state, reward, is_terminal)."""

    @abstractmethod
    def legal_actions(self, state: GameState) -> np.ndarray:
        """Boolean mask of length num_actions."""

    @abstractmethod
    def current_player(self, state: GameState) -> int:
        """0 for single-player; 0 or 1 for two-player."""

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool:
        ...

    @abstractmethod
    def blank_state(self) -> GameState:
        """A dummy state whose observable() is used as left-padding when k < q."""

    @abstractmethod
    def render_frame(self, surface: Any, state: GameState) -> None:
        """Draw the state onto a pygame.Surface. The AI core never calls this;
        PygameRenderer does."""

    def render_ascii(self, state: GameState) -> str:
        """Optional fallback for headless logs. Override for human-readable output."""
        return str(state.observable().shape)
