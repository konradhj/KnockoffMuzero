"""TicTacToe: 2-player terminal game, used to prove the AI core is game-agnostic.

Channels (state_shape = (3, 3, 3)):
  [:, :, 0] = X marks (current player's mark from player-0's view)
  [:, :, 1] = O marks
  [:, :, 2] = whose-turn plane (all 1.0 if it's X to move, else 0.0)
"""
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import SimWorld


_WIN_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]


@dataclass
class TicTacToeState:
    # board: int array (3,3) in {0=empty, 1=X, 2=O}
    board: np.ndarray
    to_move: int  # 0 = X, 1 = O

    def observable(self) -> np.ndarray:
        obs = np.zeros((3, 3, 3), dtype=np.float32)
        obs[:, :, 0] = (self.board == 1).astype(np.float32)
        obs[:, :, 1] = (self.board == 2).astype(np.float32)
        obs[:, :, 2] = 1.0 if self.to_move == 0 else 0.0
        return obs


def _winner(board: np.ndarray) -> int:
    for line in _WIN_LINES:
        vals = [board[r, c] for (r, c) in line]
        if vals[0] != 0 and all(v == vals[0] for v in vals):
            return int(vals[0])  # 1 or 2
    return 0


class TicTacToeSimWorld(SimWorld):
    @property
    def state_shape(self) -> tuple[int, ...]:
        return (3, 3, 3)

    @property
    def num_actions(self) -> int:
        return 9

    @property
    def num_players(self) -> int:
        return 2

    @property
    def reward_range(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def initial_state(self, rng: np.random.Generator) -> TicTacToeState:
        return TicTacToeState(board=np.zeros((3, 3), dtype=np.int8), to_move=0)

    def step(self, state: TicTacToeState, action: int) -> tuple[TicTacToeState, float, bool]:
        r, c = divmod(int(action), 3)
        if state.board[r, c] != 0:
            # Illegal move: end the game with a loss for the mover.
            return state, -1.0, True
        mover = state.to_move  # 0 or 1
        mark = 1 if mover == 0 else 2
        new_board = state.board.copy()
        new_board[r, c] = mark
        win = _winner(new_board)
        is_draw = (win == 0) and bool((new_board != 0).all())
        terminal = win != 0 or is_draw
        # Reward is from the perspective of the player who JUST moved.
        if win != 0:
            reward = 1.0
        else:
            reward = 0.0
        new_to_move = 1 - mover
        new_state = TicTacToeState(board=new_board, to_move=new_to_move)
        return new_state, reward, terminal

    def legal_actions(self, state: TicTacToeState) -> np.ndarray:
        if _winner(state.board) != 0:
            return np.zeros(9, dtype=bool)
        return (state.board.flatten() == 0)

    def current_player(self, state: TicTacToeState) -> int:
        return int(state.to_move)

    def is_terminal(self, state: TicTacToeState) -> bool:
        if _winner(state.board) != 0:
            return True
        return bool((state.board != 0).all())

    def blank_state(self) -> TicTacToeState:
        return TicTacToeState(board=np.zeros((3, 3), dtype=np.int8), to_move=0)

    def render_frame(self, surface: Any, state: TicTacToeState) -> None:
        import pygame

        w, h = surface.get_size()
        cell = min(w, h) // 3
        surface.fill((245, 245, 240))
        for i in range(1, 3):
            pygame.draw.line(surface, (30, 30, 30), (i * cell, 0), (i * cell, 3 * cell), 4)
            pygame.draw.line(surface, (30, 30, 30), (0, i * cell), (3 * cell, i * cell), 4)
        for r in range(3):
            for c in range(3):
                v = state.board[r, c]
                cx = c * cell + cell // 2
                cy = r * cell + cell // 2
                if v == 1:  # X
                    off = cell // 3
                    pygame.draw.line(surface, (200, 50, 50), (cx - off, cy - off), (cx + off, cy + off), 6)
                    pygame.draw.line(surface, (200, 50, 50), (cx - off, cy + off), (cx + off, cy - off), 6)
                elif v == 2:  # O
                    pygame.draw.circle(surface, (50, 90, 200), (cx, cy), cell // 3, 6)

    def render_ascii(self, state: TicTacToeState) -> str:
        chars = {0: ".", 1: "X", 2: "O"}
        rows = ["".join(chars[int(v)] for v in row) for row in state.board]
        rows.append(f"to_move={'X' if state.to_move == 0 else 'O'}")
        return "\n".join(rows)
