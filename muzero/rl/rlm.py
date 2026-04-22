"""ReinforcementLearningManager: orchestrates the episode loop.

This is the ONLY class that imports both a SimWorld and the AI core. It
mediates by:
  - pulling observables from the last q+1 real game states
  - handing the numpy stack + legal-action mask to ASM/u-MCTS
  - sampling an action from the visit distribution
  - stepping the real SimWorld
  - appending to the Episode buffer
  - periodically calling NNM.train_step
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from configs._schema import Config
from muzero.ai.nn.manager import NeuralNetworkManager
from muzero.ai.rl.episode_buffer import EpisodeBuffer, EpisodeBuilder
from muzero.ai.search.umcts import UMCTS
from muzero.io.checkpoint import save_checkpoint
from muzero.io.logging import RunLogger
from muzero.simworlds.base import GameState, SimWorld
from muzero.viz.training_plots import TrainingDashboard


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


class ReinforcementLearningManager:
    def __init__(self, simworld: SimWorld, nnm: NeuralNetworkManager,
                 umcts: UMCTS, buffer: EpisodeBuffer, cfg: Config,
                 logger: RunLogger, dashboard: TrainingDashboard,
                 renderer=None):
        self.simworld = simworld
        self.nnm = nnm
        self.umcts = umcts
        self.buffer = buffer
        self.cfg = cfg
        self.logger = logger
        self.dashboard = dashboard
        self.renderer = renderer

        self.np_rng = np.random.default_rng(cfg.run.seed)
        self._blank_obs = simworld.blank_state().observable()
        self._q = cfg.training.q

    def _history_stack(self, states_window: deque) -> np.ndarray:
        """Pack the last q+1 observables; left-pad with blanks when short."""
        target_len = self._q + 1
        pads = target_len - len(states_window)
        stack = [self._blank_obs] * max(0, pads) + [s.observable() for s in states_window]
        return np.stack(stack, axis=0).astype(np.float32)

    # --- main driver ---
    def run(self) -> None:
        for ep_idx in range(self.cfg.training.N_e):
            self._play_episode(ep_idx)
            if (ep_idx + 1) % self.cfg.training.I_t == 0:
                self._train(ep_idx)
            if (ep_idx + 1) % self.cfg.logging.checkpoint_every_episodes == 0:
                ckpt_path = Path(self.cfg.run.checkpoint_dir) / f"ep_{ep_idx + 1:05d}.eqx"
                save_checkpoint(self.nnm, ckpt_path,
                                config_snapshot={"episode": ep_idx + 1})
                self.logger.log("checkpoint", path=str(ckpt_path))

        # Final checkpoint
        final = Path(self.cfg.run.checkpoint_dir) / "final.eqx"
        save_checkpoint(self.nnm, final,
                        config_snapshot={"episodes": self.cfg.training.N_e})
        self.logger.log("done", checkpoint=str(final))
        self.dashboard.flush()

    def _play_episode(self, ep_idx: int) -> None:
        cfg = self.cfg
        state: GameState = self.simworld.initial_state(self.np_rng)
        window: deque = deque(maxlen=self._q + 1)
        builder = EpisodeBuilder()
        total_return = 0.0
        entropies: list[float] = []
        root_values: list[float] = []
        rewards_trace: list[float] = []
        visit_spreads: list[float] = []  # max-min / M_s — 0 means fully uniform
        q_spreads: list[float] = []      # max-min Q value at root
        terminal = False

        for k in range(cfg.training.N_es):
            window.append(state)
            phi_stack = self._history_stack(window)
            legal = self.simworld.legal_actions(state)
            to_play = self.simworld.current_player(state)
            result = self.umcts.run(phi_stack, legal, to_play, self.np_rng)

            # 1) ACTION-SAMPLING distribution (behavior policy): visits^(1/T_act).
            #    Annealed so early episodes explore, later episodes exploit.
            visits = result.visit_counts.astype(np.float64)
            progress = ep_idx / max(1, cfg.training.N_e - 1)
            t_act = max(0.25, 1.0 - 0.75 * progress)
            sampling = np.power(np.maximum(visits, 1e-8), 1.0 / t_act)
            sampling = sampling / sampling.sum()
            action = int(self.np_rng.choice(self.simworld.num_actions, p=sampling))

            # 2) STORED POLICY TARGET (what NN_p learns to match).
            #    a) Temperature-sharpen visits with config T (1.0 = raw visits).
            tgt_T = cfg.training.policy_target_temperature
            target = np.power(np.maximum(visits, 1e-8), 1.0 / max(1e-3, tgt_T))
            target = target / target.sum()
            #    b) Optional Q-mix: blend in softmax(Q/T_q) so the value head can
            #       provide a non-uniform signal when visit counts are flat.
            if cfg.training.q_policy_mix > 0:
                q = result.q_values.astype(np.float64)
                # only consider actions that have been visited at least once
                mask = (visits > 0).astype(np.float64)
                if mask.sum() > 0:
                    q = q - (q * mask).max()
                    q_soft = np.exp(q / max(1e-3, cfg.training.q_policy_temperature)) * mask
                    s = q_soft.sum()
                    if s > 0:
                        q_soft = q_soft / s
                        mix = cfg.training.q_policy_mix
                        target = (1.0 - mix) * target + mix * q_soft
                        target = target / target.sum()
            target = target.astype(np.float32)

            next_state, reward, terminal = self.simworld.step(state, action)

            builder.append_step(state_obs=state.observable(), action=action,
                                reward=reward, policy=target,
                                root_value=result.root_value)
            total_return += float(reward)
            entropies.append(_entropy(result.policy))
            root_values.append(float(result.root_value))
            rewards_trace.append(float(reward))
            v = result.visit_counts
            visit_spreads.append(float(v.max() - v.min()) / max(1, v.sum()))
            qv = result.q_values
            q_spreads.append(float(qv.max() - qv.min()))

            if self.renderer is not None:
                self.renderer.render(self.simworld, next_state,
                                     info={"ep": ep_idx, "step": k,
                                           "ret": f"{total_return:.2f}"})
            if terminal:
                state = next_state
                break
            state = next_state

        builder.append_final_state(state.observable())
        self.buffer.append(builder.build(terminal=terminal))

        # Actual vs predicted value error (crude: compare to n-step return from step 0)
        if root_values:
            gamma = cfg.training.gamma
            actual0 = 0.0
            for i, r in enumerate(rewards_trace):
                actual0 += (gamma ** i) * r
            value_err = abs(root_values[0] - actual0)
        else:
            value_err = float("nan")

        mean_ent = float(np.mean(entropies)) if entropies else 0.0
        self.dashboard.add_episode(total_return=total_return,
                                   mean_entropy=mean_ent,
                                   value_err=value_err)
        vspread = float(np.mean(visit_spreads)) if visit_spreads else 0.0
        qspread = float(np.mean(q_spreads)) if q_spreads else 0.0
        self.logger.log("episode", idx=ep_idx, steps=len(rewards_trace),
                        ret=total_return, entropy=mean_ent,
                        value_err=value_err, terminal=bool(terminal),
                        visit_spread=vspread, q_spread=qspread)

    def _train(self, ep_idx: int) -> None:
        if len(self.buffer) == 0:
            return
        for _ in range(self.cfg.training.gradient_steps_per_training):
            mb = self.buffer.sample_minibatch(self.np_rng)
            metrics = self.nnm.train_step(mb)
            self.dashboard.add_train_step(metrics)
        self.logger.log("train", episode=ep_idx + 1, **metrics)
        if ((ep_idx + 1) // self.cfg.training.I_t) % self.cfg.logging.plot_every_train_cycles == 0:
            self.dashboard.flush()
