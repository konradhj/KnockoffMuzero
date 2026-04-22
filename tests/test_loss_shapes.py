import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from configs._schema import LossWeights, NetworkBlockConfig, NNConfig
from muzero.ai.nn.losses import make_loss_fn
from muzero.ai.nn.networks import TriNet
from muzero.ai.types import MinibatchArrays


def test_loss_returns_scalar_and_finite_grads():
    hidden_dim = 8
    num_actions = 3
    q = 1
    w = 2
    mbs = 4
    state_shape = (4, 4, 2)

    nn_cfg = NNConfig(
        hidden_dim=hidden_dim,
        representation=NetworkBlockConfig(conv_channels=[4], conv_kernel=3, mlp_hidden=[16]),
        dynamics=NetworkBlockConfig(mlp_hidden=[16]),
        prediction=NetworkBlockConfig(mlp_hidden=[16]),
        init_scale=1.0,
    )
    trinet = TriNet(state_shape=state_shape, num_actions=num_actions,
                    q_plus_1=q + 1, cfg=nn_cfg, key=jax.random.PRNGKey(0))

    loss_fn = make_loss_fn(num_actions=num_actions, w=w,
                           weights=LossWeights(lambda_pi=1.0, lambda_v=0.25, lambda_r=1.0))

    rng = np.random.default_rng(0)
    phi = rng.standard_normal((mbs, q + 1, *state_shape)).astype(np.float32)
    actions = rng.integers(0, num_actions, size=(mbs, w)).astype(np.int32)
    target_pi = np.full((mbs, w + 1, num_actions), 1.0 / num_actions, dtype=np.float32)
    target_v = rng.standard_normal((mbs, w + 1)).astype(np.float32)
    target_r = rng.standard_normal((mbs, w)).astype(np.float32)
    mask = np.ones((mbs, w + 1), dtype=np.float32)

    mb = MinibatchArrays(
        phi_stack=jnp.asarray(phi), actions=jnp.asarray(actions),
        target_pi=jnp.asarray(target_pi), target_v=jnp.asarray(target_v),
        target_r=jnp.asarray(target_r), mask=jnp.asarray(mask),
    )

    trainable, static = eqx.partition(trinet, eqx.is_inexact_array)
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        trainable, static, mb)
    assert loss_val.shape == ()
    assert jnp.isfinite(loss_val)

    # Confirm at least some leaves have nonzero gradients.
    grad_leaves = jax.tree.leaves(grads)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in grad_leaves)
    # And that none are NaN.
    for leaf in grad_leaves:
        assert jnp.all(jnp.isfinite(leaf))
