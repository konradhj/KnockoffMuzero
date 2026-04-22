import jax
import jax.numpy as jnp

from configs._schema import NetworkBlockConfig, NNConfig
from muzero.ai.nn.networks import TriNet


def _tiny_nn_cfg(hidden_dim=16):
    return NNConfig(
        hidden_dim=hidden_dim,
        representation=NetworkBlockConfig(conv_channels=[8], conv_kernel=3, mlp_hidden=[32]),
        dynamics=NetworkBlockConfig(mlp_hidden=[32]),
        prediction=NetworkBlockConfig(mlp_hidden=[32]),
        init_scale=1.0,
    )


def test_grid_trinet_shapes():
    cfg = _tiny_nn_cfg(hidden_dim=16)
    state_shape = (4, 4, 2)
    num_actions = 3
    q_plus_1 = 3
    trinet = TriNet(state_shape=state_shape, num_actions=num_actions,
                    q_plus_1=q_plus_1, cfg=cfg, key=jax.random.PRNGKey(0))
    phi = jnp.zeros((q_plus_1, *state_shape))
    sigma = trinet.representation(phi)
    assert sigma.shape == (16,)

    a_oh = jax.nn.one_hot(jnp.asarray(0), num_actions)
    sigma_next, r = trinet.dynamics(sigma, a_oh)
    assert sigma_next.shape == (16,)
    assert r.shape == ()

    logits, v = trinet.prediction(sigma)
    assert logits.shape == (num_actions,)
    assert v.shape == ()


def test_tictactoe_trinet_shapes():
    cfg = _tiny_nn_cfg(hidden_dim=12)
    trinet = TriNet(state_shape=(3, 3, 3), num_actions=9, q_plus_1=2,
                    cfg=cfg, key=jax.random.PRNGKey(1))
    phi = jnp.zeros((2, 3, 3, 3))
    sigma = trinet.representation(phi)
    assert sigma.shape == (12,)
    a_oh = jax.nn.one_hot(jnp.asarray(4), 9)
    sigma_next, r = trinet.dynamics(sigma, a_oh)
    assert sigma_next.shape == (12,)
    logits, v = trinet.prediction(sigma)
    assert logits.shape == (9,)
