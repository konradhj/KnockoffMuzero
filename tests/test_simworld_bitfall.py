import numpy as np

from muzero.simworlds.bitfall import BitFallSimWorld


def test_initial_state_shape_and_dtype():
    sw = BitFallSimWorld(grid_rows=5, grid_cols=4, num_receptor_segments=2,
                         debris_density=0.4, horizon=10)
    rng = np.random.default_rng(0)
    s = sw.initial_state(rng)
    obs = s.observable()
    assert obs.shape == (5, 4, 2)
    assert obs.dtype == np.float32
    assert sw.state_shape == (5, 4, 2)
    assert sw.num_actions == 3
    assert sw.num_players == 1


def test_step_advances_state_and_terminates_at_horizon():
    sw = BitFallSimWorld(grid_rows=4, grid_cols=4, num_receptor_segments=1,
                         debris_density=0.3, horizon=3)
    rng = np.random.default_rng(1)
    s = sw.initial_state(rng)
    terminal = False
    steps = 0
    while not terminal and steps < 10:
        s, _r, terminal = sw.step(s, 1)
        steps += 1
    assert terminal
    assert steps == 3


def test_legal_actions_all_true():
    sw = BitFallSimWorld(grid_rows=4, grid_cols=4)
    rng = np.random.default_rng(2)
    s = sw.initial_state(rng)
    legal = sw.legal_actions(s)
    assert legal.shape == (3,)
    assert legal.all()


def test_blank_observable_is_zeros():
    sw = BitFallSimWorld(grid_rows=3, grid_cols=3)
    obs = sw.blank_state().observable()
    assert (obs == 0).all()
