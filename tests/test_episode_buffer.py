import numpy as np

from configs._schema import LossWeights, OptimizerConfig, TrainingConfig
from muzero.ai.rl.episode_buffer import EpisodeBuffer, EpisodeBuilder


def _tiny_training(q=1, w=2, mbs=3, n_step=3, gamma=0.9, buffer_capacity=10):
    return TrainingConfig(
        N_e=10, N_es=5, I_t=1, gradient_steps_per_training=1,
        mbs=mbs, q=q, w=w, gamma=gamma, n_step=n_step,
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3,
                                  weight_decay=0.0, lr_schedule="const"),
        loss_weights=LossWeights(lambda_pi=1.0, lambda_v=1.0, lambda_r=1.0),
        buffer_capacity=buffer_capacity,
    )


def test_sample_minibatch_shapes():
    cfg = _tiny_training(q=1, w=2, mbs=3)
    state_shape = (2, 2, 1)
    A = 3
    blank = np.zeros(state_shape, dtype=np.float32)
    buf = EpisodeBuffer(cfg, state_shape=state_shape, num_actions=A, blank_obs=blank)

    b = EpisodeBuilder()
    for i in range(4):
        s = np.full(state_shape, i, dtype=np.float32)
        p = np.eye(A, dtype=np.float32)[i % A]
        b.append_step(state_obs=s, action=i % A, reward=float(i), policy=p,
                      root_value=float(i * 0.1))
    b.append_final_state(np.full(state_shape, 99.0, dtype=np.float32))
    buf.append(b.build(terminal=True))

    rng = np.random.default_rng(0)
    mb = buf.sample_minibatch(rng)
    assert mb.phi_stack.shape == (3, 2, *state_shape)
    assert mb.actions.shape == (3, 2)
    assert mb.target_pi.shape == (3, 3, A)
    assert mb.target_v.shape == (3, 3)
    assert mb.target_r.shape == (3, 2)
    assert mb.mask.shape == (3, 3)
    assert (mb.mask >= 0).all() and (mb.mask <= 1).all()


def test_left_pad_blank_when_k_less_than_q():
    cfg = _tiny_training(q=2, w=1, mbs=1)
    state_shape = (2,)
    A = 2
    blank = np.full(state_shape, -99.0, dtype=np.float32)
    buf = EpisodeBuffer(cfg, state_shape=state_shape, num_actions=A, blank_obs=blank)

    b = EpisodeBuilder()
    # Episode of length 1 so k must be 0 and we need 2 pads at the front.
    b.append_step(state_obs=np.array([1.0, 2.0], dtype=np.float32), action=0,
                  reward=1.0, policy=np.array([0.5, 0.5], dtype=np.float32),
                  root_value=0.0)
    b.append_final_state(np.array([3.0, 4.0], dtype=np.float32))
    buf.append(b.build(terminal=True))

    mb = buf.sample_minibatch(np.random.default_rng(0))
    # phi_stack[0, 0] and [0, 1] should be the blank pad; [0, 2] is the real state.
    assert (mb.phi_stack[0, 0] == blank).all()
    assert (mb.phi_stack[0, 1] == blank).all()
    assert (mb.phi_stack[0, 2] == np.array([1.0, 2.0])).all()
