"""Unrolled MuZero BPTT loss.

The loss takes a minibatch of trajectories and unrolls the dynamics network w
steps ahead from the abstract state produced by the representation network.
Three heads are supervised:
  - policy : cross-entropy against u-MCTS visit distribution
  - value  : MSE against n-step bootstrapped return
  - reward : MSE against actually-observed one-step reward

Key non-obvious tricks, straight from the MuZero paper:

  * Halved-gradient sigma across the unroll. Without `sigma = 0.5*sigma +
    0.5*stop_gradient(sigma)` after each dynamics call, BPTT through w unrolls
    blows up because the same dynamics weights get stacked w times.
  * Mask is precomputed by the buffer (zero past terminal) -- the loss never
    branches on it.
  * mbs, w, num_actions baked into the closure as compile-time constants so jit
    does not retrace every step.
"""
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from configs._schema import LossWeights
from muzero.ai.nn.networks import TriNet


def _softmax_ce(logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Batched categorical CE. logits, target: (num_actions,). Returns scalar."""
    return -jnp.sum(target * jax.nn.log_softmax(logits))


def _mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    diff = pred - target
    return diff * diff


def make_loss_fn(num_actions: int, w: int, weights: LossWeights
                 ) -> Callable[[TriNet, TriNet, "MinibatchArrays"], tuple]:
    """Returns a pure function suitable for jax.value_and_grad.

    The split (trainable, static) follows the Equinox idiom: trainable holds the
    inexact arrays we differentiate; static holds everything else.
    """

    def single_example_loss(trinet: TriNet, phi_stack, actions, target_pi,
                            target_v, target_r, mask):
        # phi_stack: (q+1, *state_shape)
        # actions:   (w,)                int
        # target_pi: (w+1, num_actions)
        # target_v:  (w+1,)
        # target_r:  (w,)
        # mask:      (w+1,)
        sigma = trinet.representation(phi_stack)

        # Step 0: predict from sigma_0 (no dynamics call yet).
        logits, v = trinet.prediction(sigma)
        pi_loss = _softmax_ce(logits, target_pi[0]) * mask[0]
        v_loss = _mse(v, target_v[0]) * mask[0]
        r_loss = jnp.asarray(0.0)

        # Unroll w dynamics steps.
        for j in range(w):
            a_oh = jax.nn.one_hot(actions[j], num_actions)
            sigma, r_pred = trinet.dynamics(sigma, a_oh)
            # Halved-gradient trick: scale gradient through the unrolled state
            # by 0.5 to prevent exploding gradients from stacked dynamics calls.
            sigma = 0.5 * sigma + 0.5 * jax.lax.stop_gradient(sigma)

            logits, v = trinet.prediction(sigma)
            m = mask[j + 1]
            pi_loss = pi_loss + _softmax_ce(logits, target_pi[j + 1]) * m
            v_loss = v_loss + _mse(v, target_v[j + 1]) * m
            r_loss = r_loss + _mse(r_pred, target_r[j]) * m

        return pi_loss, v_loss, r_loss

    def batch_loss(trainable: TriNet, static: TriNet, mb) -> tuple[jnp.ndarray, dict]:
        trinet = eqx.combine(trainable, static)
        pi_l, v_l, r_l = jax.vmap(
            lambda phi, a, tp, tv, tr, m: single_example_loss(trinet, phi, a, tp, tv, tr, m)
        )(mb.phi_stack, mb.actions, mb.target_pi, mb.target_v, mb.target_r, mb.mask)

        pi_mean = jnp.mean(pi_l)
        v_mean = jnp.mean(v_l)
        r_mean = jnp.mean(r_l)
        total = (weights.lambda_pi * pi_mean
                 + weights.lambda_v * v_mean
                 + weights.lambda_r * r_mean)
        metrics = {
            "loss": total,
            "loss_pi": pi_mean,
            "loss_v": v_mean,
            "loss_r": r_mean,
        }
        return total, metrics

    return batch_loss
