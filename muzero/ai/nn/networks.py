"""The three MuZero networks wrapped in a single Equinox Module (TriNet).

Game info enters here ONLY through two scalars: state_shape and num_actions.
The AI core never imports anything from muzero/simworlds/.
"""
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from configs._schema import NetworkBlockConfig, NNConfig


def _activation(name: str) -> Callable:
    return {"relu": jax.nn.relu, "tanh": jnp.tanh, "gelu": jax.nn.gelu}[name]


class _MLP(eqx.Module):
    layers: list
    act: Callable = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int,
                 activation: str, key, init_scale: float = 1.0):
        sizes = [in_dim, *hidden, out_dim]
        keys = jax.random.split(key, len(sizes) - 1)
        layers = []
        for i, k in enumerate(keys):
            layer = eqx.nn.Linear(sizes[i], sizes[i + 1], key=k)
            # Scale weights by init_scale for stability.
            layer = eqx.tree_at(
                lambda L: L.weight, layer,
                layer.weight * init_scale,
            )
            layers.append(layer)
        self.layers = layers
        self.act = _activation(activation)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class _ConvTrunk(eqx.Module):
    convs: list
    act: Callable = eqx.field(static=True)

    def __init__(self, in_channels: int, channels: list[int], kernel: int,
                 activation: str, key):
        keys = jax.random.split(key, len(channels))
        in_c = in_channels
        convs = []
        for out_c, k in zip(channels, keys):
            convs.append(eqx.nn.Conv2d(in_c, out_c, kernel_size=kernel, padding=kernel // 2, key=k))
            in_c = out_c
        self.convs = convs
        self.act = _activation(activation)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (C, H, W)
        for c in self.convs:
            x = self.act(c(x))
        return x


class NNRepresentation(eqx.Module):
    """phi_stack -> sigma.

    Input: (q+1, *state_shape)  float32
    Output: (hidden_dim,)
    """
    is_grid: bool = eqx.field(static=True)
    state_shape: tuple = eqx.field(static=True)
    q_plus_1: int = eqx.field(static=True)
    conv: _ConvTrunk | None
    mlp: _MLP

    def __init__(self, state_shape: tuple[int, ...], q_plus_1: int, hidden_dim: int,
                 cfg: NetworkBlockConfig, key, init_scale: float):
        self.state_shape = state_shape
        self.q_plus_1 = q_plus_1
        # Treat 3-tuple state_shape (H, W, C) as a grid; anything else as a vector.
        self.is_grid = (len(state_shape) == 3)
        k1, k2 = jax.random.split(key, 2)
        if self.is_grid:
            H, W, C = state_shape
            in_c = C * q_plus_1
            self.conv = _ConvTrunk(in_c, cfg.conv_channels, cfg.conv_kernel,
                                   cfg.activation, key=k1)
            # Compute post-conv flat dim via a dummy forward so kernel/padding
            # combinations that change spatial dims still work.
            dummy = jnp.zeros((in_c, H, W))
            flat = int(self.conv(dummy).size)
            self.mlp = _MLP(flat, cfg.mlp_hidden, hidden_dim, cfg.activation,
                            key=k2, init_scale=init_scale)
        else:
            self.conv = None
            flat = int(q_plus_1 * int(jnp.prod(jnp.array(state_shape))))
            self.mlp = _MLP(flat, cfg.mlp_hidden, hidden_dim, cfg.activation,
                            key=k2, init_scale=init_scale)

    def __call__(self, phi: jnp.ndarray) -> jnp.ndarray:
        if self.is_grid:
            # phi: (q+1, H, W, C) -> (C*(q+1), H, W)
            H, W, C = self.state_shape
            x = jnp.transpose(phi, (0, 3, 1, 2))  # (q+1, C, H, W)
            x = x.reshape(self.q_plus_1 * C, H, W)
            x = self.conv(x)
            x = x.reshape(-1)
        else:
            x = phi.reshape(-1)
        x = self.mlp(x)
        # Bound sigma via tanh for numerical stability across the unroll WITHOUT
        # collapsing cross-state variance (min-max normalization did that — see
        # experiments.md run 005 probe).
        return jnp.tanh(x)


class NNDynamics(eqx.Module):
    """(sigma, action_onehot) -> (sigma', r_pred)."""
    hidden_dim: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    trunk: _MLP
    state_head: eqx.nn.Linear
    reward_head: eqx.nn.Linear

    def __init__(self, hidden_dim: int, num_actions: int,
                 cfg: NetworkBlockConfig, key, init_scale: float):
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        k1, k2, k3 = jax.random.split(key, 3)
        in_dim = hidden_dim + num_actions
        out_dim = cfg.mlp_hidden[-1] if cfg.mlp_hidden else in_dim
        if cfg.mlp_hidden:
            self.trunk = _MLP(in_dim, cfg.mlp_hidden[:-1], out_dim, cfg.activation,
                              key=k1, init_scale=init_scale)
        else:
            # Identity-ish passthrough
            self.trunk = _MLP(in_dim, [], in_dim, cfg.activation, key=k1, init_scale=init_scale)
            out_dim = in_dim
        self.state_head = eqx.nn.Linear(out_dim, hidden_dim, key=k2)
        self.reward_head = eqx.nn.Linear(out_dim, 1, key=k3)

    def __call__(self, sigma: jnp.ndarray, action_onehot: jnp.ndarray
                 ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([sigma, action_onehot], axis=-1)
        x = self.trunk(x)
        sigma_next = jnp.tanh(self.state_head(x))
        r_pred = self.reward_head(x).squeeze(-1)
        return sigma_next, r_pred


class NNPrediction(eqx.Module):
    """sigma -> (policy_logits, value)."""
    num_actions: int = eqx.field(static=True)
    trunk: _MLP
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(self, hidden_dim: int, num_actions: int,
                 cfg: NetworkBlockConfig, key, init_scale: float):
        self.num_actions = num_actions
        k1, k2, k3 = jax.random.split(key, 3)
        out_dim = cfg.mlp_hidden[-1] if cfg.mlp_hidden else hidden_dim
        if cfg.mlp_hidden:
            self.trunk = _MLP(hidden_dim, cfg.mlp_hidden[:-1], out_dim, cfg.activation,
                              key=k1, init_scale=init_scale)
        else:
            self.trunk = _MLP(hidden_dim, [], hidden_dim, cfg.activation, key=k1, init_scale=init_scale)
            out_dim = hidden_dim
        self.policy_head = eqx.nn.Linear(out_dim, num_actions, key=k2)
        self.value_head = eqx.nn.Linear(out_dim, 1, key=k3)

    def __call__(self, sigma: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.trunk(sigma)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class TriNet(eqx.Module):
    representation: NNRepresentation
    dynamics: NNDynamics
    prediction: NNPrediction

    def __init__(self, state_shape: tuple[int, ...], num_actions: int,
                 q_plus_1: int, cfg: NNConfig, key):
        kr, kd, kp = jax.random.split(key, 3)
        self.representation = NNRepresentation(state_shape, q_plus_1, cfg.hidden_dim,
                                               cfg.representation, kr, cfg.init_scale)
        self.dynamics = NNDynamics(cfg.hidden_dim, num_actions,
                                   cfg.dynamics, kd, cfg.init_scale)
        self.prediction = NNPrediction(cfg.hidden_dim, num_actions,
                                       cfg.prediction, kp, cfg.init_scale)
