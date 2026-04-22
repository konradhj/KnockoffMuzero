"""NeuralNetworkManager: owns the TriNet + optimizer, exposes jit'd forward
methods for u-MCTS, and runs one BPTT training step per call to train_step.

Single point where optax touches the codebase.
"""
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from configs._schema import NNConfig, OptimizerConfig, TrainingConfig
from muzero.ai.nn.losses import make_loss_fn
from muzero.ai.nn.networks import TriNet
from muzero.ai.types import MinibatchArrays


def _build_optimizer(cfg: OptimizerConfig) -> optax.GradientTransformation:
    if cfg.name == "adamw":
        return optax.adamw(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.name == "adam":
        return optax.adam(learning_rate=cfg.learning_rate)
    raise ValueError(f"Unknown optimizer: {cfg.name}")


class NeuralNetworkManager:
    def __init__(self, nn_cfg: NNConfig, training_cfg: TrainingConfig,
                 state_shape: tuple[int, ...], num_actions: int, seed: int = 0):
        self.nn_cfg = nn_cfg
        self.training_cfg = training_cfg
        self.state_shape = tuple(state_shape)
        self.num_actions = int(num_actions)
        self.q_plus_1 = training_cfg.q + 1
        self.w = training_cfg.w

        key = jax.random.PRNGKey(seed)
        self.trinet = TriNet(state_shape=self.state_shape, num_actions=self.num_actions,
                             q_plus_1=self.q_plus_1, cfg=nn_cfg, key=key)
        self.optimizer = _build_optimizer(training_cfg.optimizer)
        self.opt_state = self.optimizer.init(eqx.filter(self.trinet, eqx.is_inexact_array))

        # Build the jit'd loss + grad closure with w and num_actions as static ints.
        loss_fn = make_loss_fn(num_actions=self.num_actions, w=self.w,
                               weights=training_cfg.loss_weights)
        self._loss_fn = loss_fn

        @eqx.filter_jit
        def train_step(trinet, opt_state, mb):
            trainable, static = eqx.partition(trinet, eqx.is_inexact_array)
            (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                trainable, static, mb
            )
            updates, opt_state = self.optimizer.update(
                grads, opt_state, trainable
            )
            trainable = eqx.apply_updates(trainable, updates)
            new_trinet = eqx.combine(trainable, static)
            return new_trinet, opt_state, metrics

        self._train_step = train_step

        # Forward inference wrappers (jit'd once).
        @eqx.filter_jit
        def _represent(trinet, phi):
            return trinet.representation(phi)

        @eqx.filter_jit
        def _dynamics(trinet, sigma, a_oh):
            return trinet.dynamics(sigma, a_oh)

        @eqx.filter_jit
        def _predict(trinet, sigma):
            logits, v = trinet.prediction(sigma)
            return jax.nn.softmax(logits), v

        self._represent = _represent
        self._dynamics = _dynamics
        self._predict = _predict

    # --- forward methods used by u-MCTS (return numpy so tree code has no async barriers) ---
    def represent(self, phi_stack: np.ndarray) -> np.ndarray:
        sigma = self._represent(self.trinet, jnp.asarray(phi_stack, dtype=jnp.float32))
        return np.asarray(sigma)

    def dynamics(self, sigma: np.ndarray, action: int) -> tuple[np.ndarray, float]:
        a_oh = jax.nn.one_hot(jnp.asarray(action), self.num_actions)
        sigma_j = jnp.asarray(sigma, dtype=jnp.float32)
        sigma_next, r = self._dynamics(self.trinet, sigma_j, a_oh)
        return np.asarray(sigma_next), float(r)

    def predict(self, sigma: np.ndarray) -> tuple[np.ndarray, float]:
        sigma_j = jnp.asarray(sigma, dtype=jnp.float32)
        probs, v = self._predict(self.trinet, sigma_j)
        return np.asarray(probs), float(v)

    # --- training ---
    def train_step(self, mb: MinibatchArrays) -> dict[str, float]:
        mb_j = MinibatchArrays(
            phi_stack=jnp.asarray(mb.phi_stack, dtype=jnp.float32),
            actions=jnp.asarray(mb.actions, dtype=jnp.int32),
            target_pi=jnp.asarray(mb.target_pi, dtype=jnp.float32),
            target_v=jnp.asarray(mb.target_v, dtype=jnp.float32),
            target_r=jnp.asarray(mb.target_r, dtype=jnp.float32),
            mask=jnp.asarray(mb.mask, dtype=jnp.float32),
        )
        self.trinet, self.opt_state, metrics = self._train_step(self.trinet, self.opt_state, mb_j)
        return {k: float(v) for k, v in metrics.items()}

    # --- serialization ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(path, self.trinet)

    def load(self, path: str | Path) -> None:
        self.trinet = eqx.tree_deserialise_leaves(Path(path), self.trinet)
