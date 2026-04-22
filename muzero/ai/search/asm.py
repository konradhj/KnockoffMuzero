"""AbstractStateManager: the ONLY interface u-MCTS uses to talk to the neural
nets. Keeps the search code game-agnostic -- it never sees a GameState.
"""
from __future__ import annotations

import numpy as np

from muzero.ai.nn.manager import NeuralNetworkManager


class AbstractStateManager:
    def __init__(self, nnm: NeuralNetworkManager, num_actions: int, num_players: int):
        self._nnm = nnm
        self.num_actions = int(num_actions)
        self.num_players = int(num_players)

    def root_from_game_states(self, phi_stack: np.ndarray) -> np.ndarray:
        """Map q+1 observable game states to the root abstract state."""
        return self._nnm.represent(phi_stack)

    def child(self, sigma: np.ndarray, action: int) -> tuple[np.ndarray, float]:
        """One step of learned dynamics."""
        return self._nnm.dynamics(sigma, action)

    def policy_value(self, sigma: np.ndarray) -> tuple[np.ndarray, float]:
        """Policy (softmaxed) and scalar value estimate from NN_p."""
        return self._nnm.predict(sigma)
