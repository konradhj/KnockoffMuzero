"""CLI entry point.

Usage:
  python -m muzero.main --config configs/bitfall.yaml
  python -m muzero.main --config configs/bitfall.yaml --mode play --checkpoint <path>
  python -m muzero.main --config configs/bitfall.yaml --mode demo --checkpoint <path>
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from muzero.ai.nn.manager import NeuralNetworkManager
from muzero.ai.rl.episode_buffer import EpisodeBuffer
from muzero.rl.rlm import ReinforcementLearningManager
from muzero.ai.search.asm import AbstractStateManager
from muzero.ai.search.umcts import UMCTS
from muzero.config import load_config
from muzero.io.logging import RunLogger
from muzero.simworlds import build_simworld
from muzero.viz.training_plots import TrainingDashboard


def _build_all(cfg_path: str):
    cfg = load_config(cfg_path)
    simworld = build_simworld(cfg.game)
    nnm = NeuralNetworkManager(cfg.nn, cfg.training,
                               state_shape=simworld.state_shape,
                               num_actions=simworld.num_actions,
                               seed=cfg.run.seed)
    asm = AbstractStateManager(nnm, num_actions=simworld.num_actions,
                               num_players=simworld.num_players)
    umcts = UMCTS(asm, cfg.umcts, gamma=cfg.training.gamma)
    blank_obs = simworld.blank_state().observable()
    buffer = EpisodeBuffer(cfg.training, state_shape=simworld.state_shape,
                           num_actions=simworld.num_actions,
                           blank_obs=blank_obs)
    logger = RunLogger(cfg.run.log_dir, to_jsonl=cfg.logging.log_to_jsonl)
    dashboard = TrainingDashboard(cfg.run.log_dir)
    return cfg, simworld, nnm, asm, umcts, buffer, logger, dashboard


def _make_renderer(cfg, simworld):
    if not cfg.viz.pygame_enabled:
        return None
    # Initialize pygame BEFORE first JAX forward pass to avoid SDL/GPU context
    # clashes (esp. on macOS). The nnm has already been built, but JAX hasn't
    # actually traced anything yet if we haven't called forward methods.
    from muzero.viz.pygame_renderer import PygameRenderer
    return PygameRenderer(cfg.viz, state_shape=simworld.state_shape)


def cmd_train(cfg_path: str) -> None:
    cfg, simworld, nnm, asm, umcts, buffer, logger, dashboard = _build_all(cfg_path)
    renderer = _make_renderer(cfg, simworld)
    rlm = ReinforcementLearningManager(simworld=simworld, nnm=nnm, umcts=umcts,
                                       buffer=buffer, cfg=cfg,
                                       logger=logger, dashboard=dashboard,
                                       renderer=renderer)
    rlm.run()
    if renderer is not None:
        renderer.close()


def cmd_play(cfg_path: str, checkpoint: str, num_games: int = 3) -> None:
    """Play using just the actor (NN_r -> NN_p), no tree search -- as per the PDF."""
    cfg, simworld, nnm, asm, umcts, buffer, logger, dashboard = _build_all(cfg_path)
    nnm.load(checkpoint)

    # Force the renderer on for play mode.
    cfg.viz.pygame_enabled = True
    renderer = _make_renderer(cfg, simworld)

    rng = np.random.default_rng(cfg.run.seed + 1)
    q = cfg.training.q
    blank = simworld.blank_state()
    for g in range(num_games):
        state = simworld.initial_state(rng)
        window: deque = deque(maxlen=q + 1)
        total = 0.0
        for k in range(cfg.training.N_es):
            window.append(state)
            pads = (q + 1) - len(window)
            stack = [blank.observable()] * max(0, pads) + [s.observable() for s in window]
            phi_stack = np.stack(stack, axis=0).astype(np.float32)
            sigma = nnm.represent(phi_stack)
            probs, _ = nnm.predict(sigma)
            legal = simworld.legal_actions(state).astype(np.float32)
            probs = probs * legal
            if probs.sum() <= 0:
                probs = legal / max(1, legal.sum())
            else:
                probs = probs / probs.sum()
            action = int(rng.choice(simworld.num_actions, p=probs))
            state, reward, terminal = simworld.step(state, action)
            total += float(reward)
            if renderer is not None:
                renderer.render(simworld, state,
                                info={"game": g, "step": k, "ret": f"{total:.2f}"})
            if terminal:
                break
        logger.log("play", game=g, total_return=total)
    if renderer is not None:
        renderer.close()


def cmd_demo(cfg_path: str, checkpoint: str) -> None:
    """Run ONE u-MCTS search from an initial state and render the resulting tree."""
    cfg, simworld, nnm, asm, umcts, buffer, logger, dashboard = _build_all(cfg_path)
    nnm.load(checkpoint)

    rng = np.random.default_rng(cfg.run.seed + 2)
    state = simworld.initial_state(rng)
    q = cfg.training.q
    blank = simworld.blank_state()
    stack = [blank.observable()] * q + [state.observable()]
    phi_stack = np.stack(stack, axis=0).astype(np.float32)
    legal = simworld.legal_actions(state)

    # Re-run internal search but retain the tree.
    from muzero.ai.search.node import UMCTSNode
    sigma_root = asm.root_from_game_states(phi_stack)
    root = UMCTSNode(sigma=sigma_root, to_play=simworld.current_player(state),
                     legal_mask=legal.astype(bool))
    umcts._expand(root, rng=rng, add_root_noise=True)
    for _ in range(cfg.umcts.M_s):
        leaf, path, depth = umcts._descend(root)
        if not leaf.is_expanded:
            umcts._expand(leaf, rng=rng, add_root_noise=False)
        G_leaf = umcts._rollout(leaf, remaining_depth=max(0, cfg.umcts.d_max - depth), rng=rng)
        umcts._backpropagate(path, G_leaf)

    from muzero.viz.tree_viewer import draw_tree
    out = Path(cfg.run.log_dir) / "demo_tree.png"
    draw_tree(root, out, max_depth=2)
    logger.log("demo_tree", path=str(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["train", "play", "demo"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-games", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = args.mode or cfg.run.mode

    if mode == "train":
        cmd_train(args.config)
    elif mode == "play":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for play mode")
        cmd_play(args.config, args.checkpoint, num_games=args.num_games)
    elif mode == "demo":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for demo mode")
        cmd_demo(args.config, args.checkpoint)
    else:
        raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
