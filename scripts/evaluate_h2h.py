"""Head-to-head evaluator for 2-player games.

Pits the trained agent against a random opponent over many games, alternating
which side the agent plays. Reports win/draw/loss counts from the agent's
perspective — the honest measure of "how good is this player?".

Example:
  python scripts/evaluate_h2h.py \
      --config configs/tictactoe.yaml \
      --checkpoint checkpoints/tictactoe_run_001/final.eqx \
      --games 200
"""
from __future__ import annotations

import argparse
import sys
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


def play_game(simworld, nnm, umcts, cfg, agent_policy: str,
              agent_is_player: int, rng: np.random.Generator):
    """Play one 2-player game. Returns +1 if agent won, 0 if draw, -1 if lost."""
    q = cfg.training.q
    blank_obs = simworld.blank_state().observable()
    state = simworld.initial_state(rng)
    window: deque = deque(maxlen=q + 1)
    last_reward_for_agent = 0.0

    while not simworld.is_terminal(state):
        window.append(state)
        mover = simworld.current_player(state)
        legal = simworld.legal_actions(state)
        legal_idx = np.where(legal)[0]
        if legal_idx.size == 0:
            break

        if mover == agent_is_player:
            phi = _stack(window, q, blank_obs)
            if agent_policy == "actor":
                sigma = nnm.represent(phi)
                probs, _ = nnm.predict(sigma)
                probs = probs * legal.astype(np.float32)
                if probs.sum() <= 0:
                    probs = legal.astype(np.float32) / legal.sum()
                probs = probs / probs.sum()
                action = int(np.argmax(probs))
            elif agent_policy == "mcts":
                to_play = simworld.current_player(state)
                result = umcts.run(phi, legal, to_play, rng)
                action = int(np.argmax(result.visit_counts))
            else:
                raise ValueError(agent_policy)
        else:
            # random opponent
            action = int(rng.choice(legal_idx))

        state, reward, _ = simworld.step(state, action)
        # reward is from perspective of the player who just moved
        if mover == agent_is_player:
            last_reward_for_agent = reward

    # Terminal state reached. Decide outcome from the agent's perspective.
    # For TTT: reward==1.0 when mover won, 0.0 if draw.
    # If the agent was the mover of the winning move, last_reward_for_agent == 1.0.
    # If the opponent made the winning move, the agent lost (opponent's reward was 1).
    # We don't see opponent's reward directly, so we must infer from the final board.
    # Simple rule: if the game ended with reward 1 for the last mover, and that
    # mover is the agent -> +1; else -> -1. Draw -> 0.
    # Since `state` after the last step is the NEW state, look at its observables.

    # Implementation detail: rely on current_player(pre-step) via last_reward.
    # last_reward_for_agent captures the case where the agent made the final move.
    # For the opponent-makes-final-move case: if agent never received reward == 1.0,
    # but episode ended, either opponent won (-1) or it was a draw (0).
    # A cheap tell: ask the simworld if the terminal state is a win for anyone.
    # TicTacToe exposes this via step() return signals, which we've already used.

    # Simpler robust approach: rerun legal_actions on final state, if it's all
    # zeros and nobody just won, it's a draw. To know who won, we'd need a
    # winner() method. Workaround: take the last reward that was observed by
    # the LAST MOVER. If the last mover is the agent and reward==1, agent won.
    # If the last mover is the opponent and *its* reward==1 (we can't see it),
    # agent lost. If nobody won, draw. Expose winner via observable inspection:
    obs = state.observable()
    # TTT observable: channel 0 = X marks, channel 1 = O marks. Winner detected
    # by 3-in-a-row of X or O.
    if obs.ndim == 3 and obs.shape == (3, 3, 3):
        x_win = _ttt_wins(obs[:, :, 0])
        o_win = _ttt_wins(obs[:, :, 1])
        if x_win and not o_win:
            winner = 0  # X
        elif o_win and not x_win:
            winner = 1  # O
        else:
            winner = -1  # draw or both (impossible)
    else:
        # Fallback: trust last_reward_for_agent.
        if last_reward_for_agent > 0.5:
            return +1
        return 0

    if winner == -1:
        return 0
    return +1 if winner == agent_is_player else -1


def _ttt_wins(plane: np.ndarray) -> bool:
    """True if the given marker plane has 3-in-a-row."""
    # rows / cols / diags
    for i in range(3):
        if plane[i].sum() == 3: return True
        if plane[:, i].sum() == 3: return True
    if plane[0, 0] + plane[1, 1] + plane[2, 2] == 3: return True
    if plane[0, 2] + plane[1, 1] + plane[2, 0] == 3: return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    cfg = load_config(args.config)
    simworld = build_simworld(cfg.game)
    assert simworld.num_players == 2, "head-to-head eval requires a 2-player game"
    nnm = NeuralNetworkManager(cfg.nn, cfg.training,
                               state_shape=simworld.state_shape,
                               num_actions=simworld.num_actions,
                               seed=cfg.run.seed)
    nnm.load(args.checkpoint)
    asm = AbstractStateManager(nnm, num_actions=simworld.num_actions,
                               num_players=simworld.num_players)
    umcts = UMCTS(asm, cfg.umcts, gamma=cfg.training.gamma)

    rng = np.random.default_rng(args.seed)

    print(f"{'policy':10} | {'side':>6} | {'W':>4} | {'D':>4} | {'L':>4} | {'win%':>6} | {'nolose%':>8}")
    print("-" * 60)
    for policy in ("actor", "mcts"):
        for side_name, side_idx in (("X (p0)", 0), ("O (p1)", 1)):
            outcomes = {+1: 0, 0: 0, -1: 0}
            for _ in range(args.games):
                r = play_game(simworld, nnm, umcts, cfg, policy, side_idx, rng)
                outcomes[r] += 1
            W, D, L = outcomes[+1], outcomes[0], outcomes[-1]
            n = max(1, W + D + L)
            print(f"{policy:10} | {side_name:>6} | {W:4d} | {D:4d} | {L:4d} | "
                  f"{100 * W / n:5.1f}% | {100 * (W + D) / n:7.1f}%")


if __name__ == "__main__":
    main()
