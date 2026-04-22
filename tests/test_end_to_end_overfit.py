"""The canary test: if loss doesn't drop on a single synthetic trajectory,
something is wired wrong in the BPTT loss or the optimizer. Runs in <30s."""
import numpy as np

from configs._schema import (
    LossWeights,
    NetworkBlockConfig,
    NNConfig,
    OptimizerConfig,
    TrainingConfig,
)
from muzero.ai.nn.manager import NeuralNetworkManager
from muzero.ai.rl.episode_buffer import EpisodeBuffer, EpisodeBuilder


def test_loss_drops_on_single_trajectory():
    state_shape = (3, 3, 2)
    num_actions = 3
    q, w = 1, 2
    mbs = 8

    nn_cfg = NNConfig(
        hidden_dim=16,
        representation=NetworkBlockConfig(conv_channels=[8], conv_kernel=3, mlp_hidden=[32]),
        dynamics=NetworkBlockConfig(mlp_hidden=[32]),
        prediction=NetworkBlockConfig(mlp_hidden=[32]),
        init_scale=1.0,
    )
    training = TrainingConfig(
        N_e=1, N_es=6, I_t=1, gradient_steps_per_training=1,
        mbs=mbs, q=q, w=w, gamma=0.9, n_step=4,
        optimizer=OptimizerConfig(name="adam", learning_rate=0.003,
                                  weight_decay=0.0, lr_schedule="const"),
        loss_weights=LossWeights(lambda_pi=1.0, lambda_v=1.0, lambda_r=1.0),
        buffer_capacity=4,
    )

    nnm = NeuralNetworkManager(nn_cfg, training, state_shape=state_shape,
                               num_actions=num_actions, seed=0)
    blank = np.zeros(state_shape, dtype=np.float32)
    buf = EpisodeBuffer(training, state_shape=state_shape, num_actions=num_actions,
                        blank_obs=blank)

    rng = np.random.default_rng(0)
    b = EpisodeBuilder()
    T = 8
    for i in range(T):
        s = rng.standard_normal(state_shape).astype(np.float32)
        a = int(i % num_actions)
        r = float(((-1.0) ** i) * 0.5)
        # one-hot policy target to make the task learnable
        p = np.zeros(num_actions, dtype=np.float32)
        p[a] = 1.0
        b.append_step(state_obs=s, action=a, reward=r, policy=p, root_value=0.0)
    b.append_final_state(rng.standard_normal(state_shape).astype(np.float32))
    buf.append(b.build(terminal=True))

    losses = []
    for _ in range(300):
        mb = buf.sample_minibatch(rng)
        metrics = nnm.train_step(mb)
        losses.append(metrics["loss"])

    first_avg = sum(losses[:10]) / 10
    last_avg = sum(losses[-10:]) / 10
    assert last_avg < first_avg * 0.6, (
        f"loss did not drop enough: first_avg={first_avg:.4f}, last_avg={last_avg:.4f}"
    )
