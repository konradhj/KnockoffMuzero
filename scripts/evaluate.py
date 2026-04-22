"""Baseline comparison harness.

Runs N episodes of a SimWorld under several policies and reports mean return
and std. Use this to ground-truth how good the learned agent is.

Policies:
  random     — uniform action each step
  stay       — always pick action index 1 (BitFall: "stay")
  actor      — NN_r + NN_p greedy (no MCTS), as per the PDF's "actor" section
  mcts       — full u-MCTS search (the trained agent as used in training)

Example:
  python scripts/evaluate.py --config configs/bitfall.yaml --episodes 100 \
      --checkpoint checkpoints/bitfall/final.eqx
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muzero.ai.nn.manager import NeuralNetworkManager
from muzero.ai.search.asm import AbstractStateManager
from muzero.ai.search.umcts import UMCTS
from muzero.config import load_config
from muzero.simworlds import build_simworld


def _stack(window, q, blank_obs):
    pads = (q + 1) - len(window)
    arr = [blank_obs] * max(0, pads) + [s.observable() for s in window]
    return np.stack(arr, axis=0).astype(np.float32)


def run_policy(name, cfg, simworld, nnm, umcts, num_episodes, rng):
    q = cfg.training.q
    blank_obs = simworld.blank_state().observable()
    returns = []
    lengths = []
    t0 = time.time()
    for ep in range(num_episodes):
        state = simworld.initial_state(rng)
        window: deque = deque(maxlen=q + 1)
        ret = 0.0
        steps = 0
        for k in range(cfg.training.N_es):
            window.append(state)
            legal = simworld.legal_actions(state)
            legal_idx = np.where(legal)[0]
            if name == "random":
                action = int(rng.choice(legal_idx))
            elif name == "stay":
                # action index 1 is "STAY" in BitFall. For other games fall back
                # to first legal action.
                action = 1 if 1 in legal_idx else int(legal_idx[0])
            elif name == "actor":
                phi = _stack(window, q, blank_obs)
                sigma = nnm.represent(phi)
                probs, _ = nnm.predict(sigma)
                probs = probs * legal.astype(np.float32)
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
                action = int(np.argmax(probs))
            elif name == "mcts":
                phi = _stack(window, q, blank_obs)
                to_play = simworld.current_player(state)
                result = umcts.run(phi, legal, to_play, rng)
                action = int(np.argmax(result.visit_counts))
            else:
                raise ValueError(f"unknown policy: {name}")
            state, r, terminal = simworld.step(state, action)
            ret += float(r)
            steps += 1
            if terminal:
                break
        returns.append(ret)
        lengths.append(steps)
    elapsed = time.time() - t0
    arr = np.asarray(returns)
    return {
        "policy": name,
        "episodes": num_episodes,
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "mean_steps": float(np.mean(lengths)),
        "time_s": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="required for actor/mcts policies")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policies", nargs="+",
                        default=["random", "stay", "actor", "mcts"])
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    cfg = load_config(args.config)
    simworld = build_simworld(cfg.game)
    nnm = NeuralNetworkManager(cfg.nn, cfg.training,
                               state_shape=simworld.state_shape,
                               num_actions=simworld.num_actions,
                               seed=cfg.run.seed)
    if args.checkpoint is not None:
        nnm.load(args.checkpoint)
    asm = AbstractStateManager(nnm, num_actions=simworld.num_actions,
                               num_players=simworld.num_players)
    umcts = UMCTS(asm, cfg.umcts, gamma=cfg.training.gamma)

    print(f"{'policy':10} | {'mean_ret':>8} | {'std':>6} | {'min':>6} | "
          f"{'max':>6} | {'len':>5} | {'time':>6}")
    print("-" * 72)
    for name in args.policies:
        rng = np.random.default_rng(args.seed)
        if name in ("actor", "mcts") and args.checkpoint is None:
            print(f"{name:10} | (skipped: --checkpoint required)")
            continue
        stats = run_policy(name, cfg, simworld, nnm, umcts, args.episodes, rng)
        print(f"{stats['policy']:10} | {stats['mean_return']:8.3f} | "
              f"{stats['std_return']:6.3f} | {stats['min_return']:6.2f} | "
              f"{stats['max_return']:6.2f} | {stats['mean_steps']:5.1f} | "
              f"{stats['time_s']:6.2f}s")


if __name__ == "__main__":
    main()
