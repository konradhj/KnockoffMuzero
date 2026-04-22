import numpy as np

from muzero.simworlds.tictactoe import TicTacToeSimWorld


def test_descriptors():
    sw = TicTacToeSimWorld()
    assert sw.state_shape == (3, 3, 3)
    assert sw.num_actions == 9
    assert sw.num_players == 2


def test_x_wins_top_row():
    sw = TicTacToeSimWorld()
    rng = np.random.default_rng(0)
    s = sw.initial_state(rng)
    # X at (0,0), O at (1,0), X at (0,1), O at (1,1), X at (0,2) => X wins
    moves = [0, 3, 1, 4, 2]
    terminal = False
    reward = 0.0
    for m in moves:
        s, reward, terminal = sw.step(s, m)
    assert terminal
    assert reward == 1.0  # last mover (X) won


def test_legal_actions_shrink():
    sw = TicTacToeSimWorld()
    rng = np.random.default_rng(0)
    s = sw.initial_state(rng)
    s, _, _ = sw.step(s, 4)
    legal = sw.legal_actions(s)
    assert legal.sum() == 8
    assert not legal[4]


def test_current_player_alternates():
    sw = TicTacToeSimWorld()
    rng = np.random.default_rng(0)
    s = sw.initial_state(rng)
    assert sw.current_player(s) == 0
    s, _, _ = sw.step(s, 0)
    assert sw.current_player(s) == 1
