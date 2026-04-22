"""Quick probe: load a checkpoint and see whether the representation network
produces meaningfully different sigma (and Q) for obviously different states.

If sigma is approximately the same across radically different game states, the
representation is collapsed — training will never produce a useful policy or Q.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.ai.nn.manager import NeuralNetworkManager
from muzero.config import load_config
from muzero.simworlds import build_simworld


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/bitfall_005.yaml"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/bitfall_run_005/final.eqx"

    cfg = load_config(cfg_path)
    sw = build_simworld(cfg.game)
    nnm = NeuralNetworkManager(cfg.nn, cfg.training, state_shape=sw.state_shape,
                               num_actions=sw.num_actions, seed=cfg.run.seed)
    nnm.load(ckpt)

    rng = np.random.default_rng(0)
    # Build a handful of obviously-different observables by stacking manual states.
    q = cfg.training.q
    blank_obs = sw.blank_state().observable()

    def make_stack(states):
        pads = (q + 1) - len(states)
        arr = [blank_obs] * max(0, pads) + [s.observable() for s in states]
        return np.stack(arr, axis=0).astype(np.float32)

    # Different receptor + debris configurations.
    states = [sw.initial_state(np.random.default_rng(s)) for s in range(5)]
    sigmas = np.stack([nnm.represent(make_stack([s])) for s in states], axis=0)
    print("sigma_mean_pairwise_dist:", np.mean(
        [np.linalg.norm(sigmas[i] - sigmas[j])
         for i in range(len(sigmas)) for j in range(len(sigmas)) if i < j]))
    print("sigma_std_per_dim (mean):", float(sigmas.std(axis=0).mean()))
    print("sigma_range:", float(sigmas.min()), "to", float(sigmas.max()))

    # Q across actions from a single state.
    sigma = sigmas[0]
    qs = []
    for a in range(sw.num_actions):
        sigma_next, r_pred = nnm.dynamics(sigma, a)
        _, v = nnm.predict(sigma_next)
        qs.append(float(r_pred) + cfg.training.gamma * float(v))
    qs = np.array(qs)
    print(f"Q across {sw.num_actions} actions: {qs}")
    print(f"Q spread: {qs.max() - qs.min():.4f}")

    # Same for the policy head directly on sigma.
    probs, v = nnm.predict(sigma)
    print(f"NN_p direct policy: {probs}")
    print(f"NN_p direct value:  {v}")


if __name__ == "__main__":
    main()
