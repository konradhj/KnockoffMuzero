"""u-MCTS: Monte Carlo Tree Search over abstract states.

Plain Python object tree (dataclasses). At M_s up to a few hundred and
branching <= |A|, tree traversal is cheap; NN calls dominate.

Flow of one simulation:
  1. Descend from root picking edges by UCB until an unexpanded node (leaf).
  2. Expand the leaf: create children via NN_d; priors/value via NN_p.
  3. From the leaf, pick a random child c*, roll out to a depth bounded by
     d_max using NN_p (action sampling) and NN_d (transitions), accumulating
     rewards. Bootstrap terminal contribution from NN_p.value at the tail.
  4. Back-propagate the accumulated return up the path to the root. The only
     2-player-aware line in the AI core lives in the sign-flip inside
     _backpropagate (and _rollout).

Value/perspective convention (zero-sum 2-player):
  - edge.reward is stored as the reward received by the player who chose the
    edge (parent.to_play).
  - A G computed at some node is from the perspective of that node's to_play
    (the player about to move).
  - Crossing an edge upward flips sign: G_parent = r_edge + gamma * (-G_child).
  - Edge Q values are stored from the parent's perspective (the chooser), so
    UCB's naive argmax is correct.
  - In 1-player games the flip collapses to identity.
"""
from __future__ import annotations

import math

import numpy as np

from configs._schema import UMCTSConfig
from muzero.ai.search.asm import AbstractStateManager
from muzero.ai.search.node import UMCTSEdge, UMCTSNode
from muzero.ai.types import SearchResult


class UMCTS:
    def __init__(self, asm: AbstractStateManager, cfg: UMCTSConfig, gamma: float):
        self.asm = asm
        self.cfg = cfg
        self.gamma = float(gamma)
        self.num_actions = asm.num_actions
        self.num_players = asm.num_players

    def run(self, phi_stack: np.ndarray, root_legal_mask: np.ndarray,
            to_play: int, rng: np.random.Generator) -> SearchResult:
        sigma_root = self.asm.root_from_game_states(phi_stack)
        root = UMCTSNode(sigma=sigma_root, to_play=int(to_play),
                         legal_mask=root_legal_mask.astype(bool))
        self._expand(root, rng=rng, add_root_noise=True)

        for _ in range(self.cfg.M_s):
            leaf, path_edges, depth = self._descend(root)
            if not leaf.is_expanded:
                self._expand(leaf, rng=rng, add_root_noise=False)
            G_leaf = self._rollout(leaf,
                                   remaining_depth=max(0, self.cfg.d_max - depth),
                                   rng=rng)
            self._backpropagate(path_edges, G_leaf)

        visit_counts = np.array([e.visit_count for e in root.edges], dtype=np.int64)
        q_values = np.array([e.q_value() for e in root.edges], dtype=np.float32)
        total = int(visit_counts.sum())
        policy = visit_counts.astype(np.float32) / max(1, total)
        if total > 0:
            root_value = float(
                sum(e.q_value() * e.visit_count for e in root.edges) / total
            )
        else:
            _, v0 = self.asm.policy_value(sigma_root)
            root_value = float(v0)
        return SearchResult(visit_counts=visit_counts, policy=policy,
                            root_value=root_value, q_values=q_values)

    # --- tree policy ---
    def _descend(self, root: UMCTSNode) -> tuple[UMCTSNode, list[UMCTSEdge], int]:
        node = root
        path: list[UMCTSEdge] = []
        depth = 0
        while node.is_expanded and depth < self.cfg.d_max:
            edge = self._select_edge(node)
            path.append(edge)
            assert edge.child is not None, "expanded node must have children"
            node = edge.child
            depth += 1
        return node, path, depth

    def _select_edge(self, node: UMCTSNode) -> UMCTSEdge:
        best = None
        best_score = -math.inf
        total_n = node.total_visit_count()
        sqrt_total = math.sqrt(max(1, total_n))
        for e in node.edges:
            if node.legal_mask is not None and not node.legal_mask[e.action]:
                continue
            q = e.q_value()
            u = self.cfg.c_ucb * e.prior * sqrt_total / (1 + e.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best = e
        if best is None:
            # fallback: pick the most-visited legal edge or the first
            for e in node.edges:
                if node.legal_mask is None or node.legal_mask[e.action]:
                    return e
            return node.edges[0]
        return best

    # --- expansion ---
    def _expand(self, node: UMCTSNode, rng: np.random.Generator,
                add_root_noise: bool) -> None:
        if node.is_expanded:
            return
        priors, _ = self.asm.policy_value(node.sigma)
        priors = np.asarray(priors, dtype=np.float64)
        if node.legal_mask is not None:
            priors = priors * node.legal_mask
            if priors.sum() <= 0:
                priors = node.legal_mask.astype(np.float64)
            priors = priors / priors.sum()

        if (add_root_noise and self.cfg.dirichlet_alpha is not None
                and self.cfg.dirichlet_alpha > 0):
            noise = rng.dirichlet([self.cfg.dirichlet_alpha] * self.num_actions)
            frac = self.cfg.dirichlet_frac
            priors = (1 - frac) * priors + frac * noise
            if node.legal_mask is not None:
                priors = priors * node.legal_mask
                if priors.sum() > 0:
                    priors = priors / priors.sum()

        for a in range(self.num_actions):
            if node.legal_mask is not None and not node.legal_mask[a]:
                # still create the edge so indexing matches, but flag it dead.
                node.edges.append(UMCTSEdge(action=a, prior=0.0))
                continue
            sigma_next, r = self.asm.child(node.sigma, a)
            next_to_play = (1 - node.to_play) if self.num_players == 2 else node.to_play
            child = UMCTSNode(sigma=sigma_next, to_play=next_to_play)
            node.edges.append(UMCTSEdge(action=a, prior=float(priors[a]),
                                        reward=float(r), child=child))
        node.is_expanded = True

    # --- rollout ---
    def _rollout(self, leaf: UMCTSNode, remaining_depth: int,
                 rng: np.random.Generator) -> float:
        """Return G at the leaf from the leaf's to_play perspective.

        Two modes:
          - rollout_enabled=False: just return NN_p.value(leaf.sigma). Best for
            stochastic games where NN_d + NN_p rollouts inject pure noise.
          - rollout_enabled=True: pick a random legal edge, roll forward
            remaining_depth steps using NN_p samples, bootstrap with NN_p.value
            at the tail, fold rewards backward (with a sign flip per step in 2P).
        """
        if not self.cfg.rollout_enabled:
            _, v = self.asm.policy_value(leaf.sigma)
            return float(v)

        legal_edges = [e for e in leaf.edges if e.child is not None]
        if not legal_edges:
            _, v = self.asm.policy_value(leaf.sigma)
            return float(v)

        first = legal_edges[rng.integers(0, len(legal_edges))]
        rewards = [first.reward]
        sigma = first.child.sigma

        if remaining_depth > 0:
            for _ in range(remaining_depth):
                probs, _ = self.asm.policy_value(sigma)
                probs = np.asarray(probs, dtype=np.float64)
                s = probs.sum()
                probs = probs / s if s > 0 else np.full(self.num_actions, 1.0 / self.num_actions)
                a = int(rng.choice(self.num_actions, p=probs))
                sigma, r = self.asm.child(sigma, a)
                rewards.append(float(r))

        _, v_tail = self.asm.policy_value(sigma)
        G = float(v_tail)
        # Walking backward. At each step, the reward is from the current mover's
        # perspective; G is currently from the NEXT state's to_play perspective.
        # In 2P, each backward step flips sign.
        for r in reversed(rewards):
            if self.num_players == 2:
                G = r + self.gamma * (-G)
            else:
                G = r + self.gamma * G
        return G


    # --- backprop ---
    def _backpropagate(self, path: list[UMCTSEdge], G_leaf: float) -> None:
        """Walk up the path. G is tracked from the current child's perspective;
        at each edge we flip (2P) and add the edge reward to get parent's view."""
        G = G_leaf
        for edge in reversed(path):
            if self.num_players == 2:
                G = edge.reward + self.gamma * (-G)
            else:
                G = edge.reward + self.gamma * G
            edge.visit_count += 1
            edge.total_value += G
