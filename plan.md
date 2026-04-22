# MuZero Knockoff — Code Design Plan

## Context

This is the IT-3105 main project: build a from-scratch MuZero knockoff that learns to play a simple arcade game by combining u-MCTS over abstract states with a trio of interlinked neural networks Ψ = (NN_r, NN_d, NN_p), trained via on-policy BPTT. The project is also graded on code structure (critical divide, one-place parameters, flexibility, transparency) per the "code expectations" PDF — failing those can cost up to 50% of the points regardless of whether the system learns.

**Confirmed choices:** JAX + Equinox; BitFall as primary game; a second SimWorld (TicTacToe) that plugs into the same AI core to demonstrate generality; Pygame for live game-state rendering; matplotlib for training curves.

Deliverable is a 10-minute educational video, but this plan covers only the **code**. The code must be demo-able (game board visible, training-progress plots available) since the video records output from this system.

---

## Directory Layout

```
KnockoffMuzero/
├── pyproject.toml
├── configs/
│   ├── bitfall.yaml                    ← single source of truth per run
│   ├── tictactoe.yaml
│   └── _schema.py                      ← typed dataclasses mirroring YAML
├── muzero/
│   ├── main.py                         ← CLI: train | play | demo
│   ├── config.py                       ← YAML → dataclasses, validated on load
│   ├── simworlds/                      ← GAME-SPECIFIC (nothing in ai/ may import from here)
│   │   ├── base.py                     ← SimWorld ABC + GameState protocol
│   │   ├── bitfall.py
│   │   └── tictactoe.py
│   ├── ai/                             ← GAME-AGNOSTIC
│   │   ├── nn/
│   │   │   ├── networks.py             ← NNRepresentation, NNDynamics, NNPrediction, TriNet (eqx.Module)
│   │   │   ├── manager.py              ← NeuralNetworkManager: owns TriNet + optax state + train_step
│   │   │   └── losses.py               ← unrolled_loss (BPTT over q+1 repr inputs, w dynamics unrolls)
│   │   ├── search/
│   │   │   ├── node.py                 ← UMCTSNode, UMCTSEdge dataclasses
│   │   │   ├── umcts.py                ← run, descend, expand, rollout, backprop
│   │   │   └── asm.py                  ← AbstractStateManager: NNM adapter exposed to u-MCTS
│   │   ├── rl/
│   │   │   ├── episode_buffer.py       ← EpisodeBuffer + minibatch sampler (q-lookback, w-rollahead, masks)
│   │   │   └── rlm.py                  ← ReinforcementLearningManager: episode loop (only class holding both SimWorld and NNM)
│   │   └── types.py
│   ├── viz/
│   │   ├── pygame_renderer.py          ← generic shell; delegates to simworld.render_frame
│   │   ├── training_plots.py           ← matplotlib: loss curves, return, value error, policy entropy
│   │   └── tree_viewer.py              ← demo mode: animates a single u-MCTS search growing
│   └── io/
│       ├── checkpoint.py               ← eqx.tree_serialise wrappers (saves config alongside)
│       └── logging.py                  ← jsonl + stdout
└── tests/                              ← see §Testing below
```

**Import invariant:** nothing under `muzero/ai/` may import from `muzero/simworlds/`. Enforced by `tests/test_import_boundary.py` (greps source). This is the "critical divide" requirement.

Dependency direction: `main → {config, simworlds, ai/rl/rlm, viz}`; `rlm → {umcts, asm, nnm, buffer}`; `umcts → {node, asm}`; `asm → nnm`; `simworlds/*` → `simworlds/base` only.

Rough scope: ~3.5–4k LOC total (AI core ~1.7k, simworlds ~0.5k, viz ~0.55k, config+io ~0.25k, tests ~0.8k).

---

## Key Classes

### SimWorld (game-specific, in its own file per game)
ABC with: `state_shape`, `num_actions`, `num_players`, `reward_range`, `initial_state(rng)`, `step(state, action) → (next, reward, terminal)`, `legal_actions(state) → bool mask`, `current_player(state)`, `is_terminal(state)`, `blank_state()` (padding when `k<q`), `render_frame(surface, state)`.

**Decision: no separate `GameStateManager` class.** The GSM duties from the PDF are all handled on SimWorld itself; a parallel class would just duplicate the interface. This is a deliberate consolidation — call it out in the video/docs so graders see it was considered.

- `BitFallSimWorld`: 1-player, 3 actions (left/stay/right), no terminal state (fixed-horizon episodes), `state_shape = (rows, cols, C)` with channels for debris + receptors.
- `TicTacToeSimWorld`: 2-player, 9 actions, terminal, `state_shape = (3,3,3)` with channels for X-marks, O-marks, whose-turn plane. The whose-turn plane is how NN_r learns turn-awareness without the AI core needing a Player abstraction.

### TriNet (`ai/nn/networks.py`)
Single `eqx.Module` containing three sub-modules:
- `NNRepresentation`: input `(q+1, *state_shape)` (channel-concatenated for grid games, flat-concatenated for vector states); output `σ ∈ ℝ^H`. Branches on `len(state_shape)` to pick CNN-prefix or MLP — purely structural, no semantic game knowledge.
- `NNDynamics`: input `(σ, onehot(a))`; output `(σ', r_pred)`. One-hot + MLP (small |A|; embeddings add no value at these sizes).
- `NNPrediction`: input `σ`; output `(policy_logits ∈ ℝ^|A|, v ∈ ℝ)`.

`hidden_dim = H` is a config scalar (e.g. 64 for BitFall, 32 for TTT). Game info enters NN construction **only** via two scalars (`state_shape`, `num_actions`) read once from SimWorld by `main.py`'s factory code.

### NeuralNetworkManager (`ai/nn/manager.py`)
Owns `TriNet`, `optax` optimizer, opt state. Exposes:
- `represent(phi_stack) → σ`, `dynamics(σ, a) → (σ', r)`, `predict(σ) → (π, v)` — single-sample, jit-wrapped, game-agnostic.
- `batch_*` versions via `jax.vmap` for minibatches.
- `train_step(minibatch) → metrics`: thin shell around `jax.value_and_grad(unrolled_loss)` + optimizer update.
- `save(path)`, `load(path)`: `eqx.tree_serialise_leaves` + saves config snapshot next to checkpoint so load validates.

### unrolled_loss (`ai/nn/losses.py`)
The single trickiest artifact. Signature:
```python
def unrolled_loss(trainable: TriNet, static: TriNet, mb: MinibatchArrays, cfg: LossConfig) -> (scalar, metrics)
```
Flow: vmap NN_r over `mb.phi_stack (B, q+1, *SS)` → `σ₀`. At step 0 and each of w unrolls, compute π,v from NN_p and accumulate head losses (policy CE, value MSE); at each unroll also compute `(σ_next, r_pred)` from NN_d and accumulate reward MSE against `mb.target_r`. Mask past-terminal entries via `mb.mask`.

**Non-obvious bits that MUST go in code comments:**
- **Halved-gradient σ trick** across each unroll: `σ = 0.5·σ + 0.5·stop_gradient(σ)` — from the MuZero paper; without it, BPTT through w unrolls explodes.
- `stop_gradient` on the bootstrap target for value.
- `mbs`, `q`, `w` must be compile-time constants baked into the closure (else jit retraces).
- Start with plain Python `for j in range(w):` (w small) rather than `jax.lax.scan`.

Loss weights `λ_π = 1.0, λ_v = 0.25, λ_r = 1.0` (MuZero paper defaults; in config).

### UMCTS + UMCTSNode + UMCTSEdge (`ai/search/`)
Plain Python dataclasses (not JAX arrays). At M_s ≤ 200 and branching ≤ |A| ≤ 9, a Python object tree is the right choice — simpler backprop-along-path, trivial to debug/visualize, NN cost dominates anyway.

`UMCTSEdge`: `action, reward (cached from NN_d at expansion), prior, visit_count, total_value, child`.
`UMCTSNode`: `sigma (numpy, post `device_get`), is_expanded, edges, parent_edge, to_play`.

`UMCTS.run(phi_stack, root_legal_mask, num_players, rng)` → `{visit_counts, root_value, root_node}`. Internals: `_descend_tree_policy` (UCB), `_expand_and_rollout` (calls NN_d for children, NN_p for priors/value, then a depth-`d_max − d` rollout using NN_p for action sampling and NN_d for transitions), `_backpropagate` (n-step return with γ, with sign-flip in 2-player mode).

### AbstractStateManager (`ai/search/asm.py`)
Thin adapter: u-MCTS never touches NNM directly. ASM accepts `phi_stack` and legal-mask as **numpy arrays** — never a `GameState`. Methods: `root_from_game_states`, `child(σ, a)`, `policy_value(σ)`, `legal_actions_at_root(mask)`, `legal_actions_in_tree()`.

### EpisodeBuffer (`ai/rl/episode_buffer.py`)
`EpisodeRecord`: numpy arrays for states `(T+1, *SS)`, actions `(T,)`, rewards `(T,)`, policies `(T, |A|)`, root_values `(T,)`, terminal flag.

`sample_minibatch(mbs, rng) → MinibatchArrays`:
- `phi_stack (mbs, q+1, *SS)` — slice `[k-q, k]`, left-pad with SimWorld blank state when `k<q`.
- `actions (mbs, w)` — slice `[k+1, k+w]`.
- `target_pi (mbs, w+1, |A|)`, `target_v (mbs, w+1)`, `target_r (mbs, w)`.
- `mask (mbs, w+1)` — 0 past terminal/end-of-episode, 1 otherwise. Zero out rather than branch.

Value target = n-step return with bootstrap from stored `v*` (u-MCTS root value at that step, not a fresh NN_p call): `target_v[b,j] = Σ γⁱ r_{k+j+i+1} + γⁿ v*_{k+j+n}`, with `n = min(n_step, T − (k+j))`.

### ReinforcementLearningManager (`ai/rl/rlm.py`)
Only class holding both SimWorld and NNM references. Mediates: calls `simworld.observable()` on the last q+1 real states, stacks, hands numpy to ASM/UMCTS; samples action from visit counts; steps SimWorld; appends to `EpisodeRecord`; every I_t episodes, runs `gradient_steps_per_training` calls of `nnm.train_step`.

### PygameRenderer (`viz/pygame_renderer.py`)
Generic shell: window, event pump, HUD (episode/step/score), enable/disable flag. Delegates drawing to `simworld.render_frame(self.screen, state)`. The only game-specific rendering code lives in each SimWorld.

---

## Critical Design Decisions (rationale captured in code comments)

- **NN_r input:** channel-concat stack for grid games, flat concat for vector. Only shape-dependent branch in AI core; purely structural.
- **Action encoding into NN_d:** one-hot concat. Clean, small, vmap-friendly. Swap for learned embedding later via same interface if action set grows.
- **Two-player handling (TicTacToe):** AI core has exactly **one** 2-player-aware line — a sign-flip in `_backpropagate` gated on `num_players == 2 and node.to_play != root.to_play`. NN_p stays player-agnostic: the "whose turn" plane inside TTT's observable encodes turn implicitly, and NN_r/NN_d learn dynamics over it naturally.
- **BitFall termination:** no terminal state; fixed-horizon `N_es` episodes; value target uses n-step bootstrap from stored `v*`. **TicTacToe:** terminal; n-step effectively unbounded; mask zeros past-terminal. Both identical in EpisodeBuffer code — behavior toggles on `is_terminal`.
- **On-policy:** target policy = u-MCTS visit distribution (stored in EpisodeRecord); behavior policy = sample from same distribution. No off-policy corrections.
- **Discounting in backprop:** iterative-from-leaf style. Leaf value `G = Σ γⁱ rollout_r[i] + γ^len v_leaf`; climbing, `G ← r_edge + γ·G`; then `edge.total_value += G; edge.visit_count += 1`.
- **Dirichlet noise at root priors** (`α=0.3, frac=0.25`): important for BitFall exploration. Wired in, toggleable from config.

---

## Config File Schema (single source of truth)

`configs/bitfall.yaml` sections — every pivotal parameter lives here, nothing else:
- `run:` name, seed, checkpoint_dir, log_dir, mode
- `game:` name (dispatched via simworlds factory) + params (grid dims, receptor count, debris density). `state_shape` / `num_actions` are **derived** from SimWorld, never set here.
- `umcts:` `M_s, d_max, c_ucb, dirichlet_alpha, dirichlet_frac, rollout_enabled`
- `nn:` `hidden_dim` + per-network `conv_channels, mlp_hidden, activation` + `init_scale`
- `training:` `N_e, N_es, I_t, gradient_steps_per_training, mbs, q, w, gamma, n_step`, optimizer (`name, learning_rate, weight_decay, lr_schedule`), `loss_weights {λ_π, λ_v, λ_r}`, `buffer_capacity`
- `logging:` plot frequency, checkpoint frequency, jsonl toggle
- `viz:` `pygame_enabled, pygame_fps, cell_size_px, window_title`

`configs/tictactoe.yaml` overrides `game.*`, `training.gamma: 1.0`, smaller `N_es`, no `n_step`. Swapping games = `--config configs/tictactoe.yaml`. Zero source edits.

A one-off `scripts/audit_config.py` greps `.py` outside `configs/` for float literals > 1.0 — catches "I hard-coded LR=1e-3 somewhere" before grading.

---

## CLI Run Modes (`muzero/main.py`)

```
python -m muzero.main --config configs/bitfall.yaml --mode train
                                                   --mode play  --checkpoint path
                                                   --mode demo  --checkpoint path
```

- `train`: episode loop, save checkpoints + training curves. pygame off by default.
- `play`: load checkpoint. Uses the "actor" from the PDF — drops NN_d; just `NN_r → NN_p → sample action`. No tree search. pygame on.
- `demo`: load checkpoint, run ONE u-MCTS move with `tree_viewer.py` animating the tree growing. This is the money shot for the video.

---

## Testing Strategy (ordered by impact, all < 10s total)

1. `test_import_boundary.py` — greps `muzero/ai/` for simworld imports. **First** line of defense for the critical-divide grade.
2. `test_networks_shapes.py` — TriNet construction + shape assertions for several `(state_shape, |A|, H)` combos.
3. `test_loss_shapes.py` — synthesize random minibatch of expected shapes; call `unrolled_loss`; assert scalar + finite grads on every TriNet leaf. Catches broadcasting without needing real episodes.
4. `test_umcts_invariants.py` — `Σ visit_counts == M_s`; every expanded node has `|A|` edges; depth ≤ d_max; hand-computed UCB; 2-player sign-flip on a hand-built tree.
5. `test_simworld_bitfall.py`, `test_simworld_tictactoe.py` — observable shape/dtype, deterministic; TTT scripted winning game; BitFall reward-sign sanity.
6. `test_episode_buffer.py` — shapes, mask past terminal, left-pad with blank.
7. `test_end_to_end_overfit.py` — 1 synthetic episode, 500 grad steps, loss must drop ≥ 1 order of magnitude. Canary for "is training wired correctly."

Explicit non-goal: unit-testing that MuZero learns BitFall. That's an integration smoke run.

---

## Scope Risks (ranked)

1. **BitFall actually learning.** Mitigate: overfit test first; start with tiny grid (4×4); log policy entropy — collapse to uniform or delta in first 100 episodes = targets broken.
2. **BPTT loss correctness.** Specific pitfalls: halved-grad σ trick omitted → divergence; slice misalignment across `target_pi / target_v / target_r` → silently learns nothing; forgetting `stop_gradient` on value bootstrap. Mitigation: write in numpy first, verify 2-step unroll by hand, then port to JAX.
3. **2-player sign flip.** Easy to get wrong; tree "kind of works" but picks losing moves. Mitigation: unit test #4.
4. **Episode buffer slicing off-by-ones** — guaranteed without explicit test of §6.
5. **JIT retracing** if `mbs/q/w` aren't static. Bake as closure constants.
6. **Pygame + JAX on macOS:** init pygame BEFORE first JAX call; `JAX_PLATFORMS=cpu` for viz modes if SDL contexts conflict.
7. **Checkpoint compatibility:** `eqx.tree_serialise` is structure-sensitive. Save config next to checkpoint, validate on load.

---

## Critical Files to Implement (highest bug density → highest impact)

- `muzero/ai/nn/losses.py` — the BPTT unrolled loss; get this right or nothing trains.
- `muzero/ai/search/umcts.py` — expansion, rollout, discounting, 2-player sign-flip.
- `muzero/ai/rl/episode_buffer.py` — slicing + masking; off-by-ones silently break training.
- `muzero/ai/rl/rlm.py` — the only class bridging SimWorld and AI core; carries the "clean divide" responsibility.
- `muzero/simworlds/base.py` — the SimWorld ABC; every generality decision collapses here.

---

## Verification (how to demonstrate the system works end-to-end)

1. `pytest tests/` — all tests green. Most importantly `test_import_boundary` (critical divide) and `test_end_to_end_overfit` (training plumbing).
2. `python scripts/audit_config.py` — zero stray literals outside `configs/`.
3. `python -m muzero.main --config configs/bitfall.yaml --mode train` — runs until `N_e`; produces `training_dashboard.png` showing: total loss decreasing, policy loss + value loss + reward loss each decreasing, mean episode return trending up, value-prediction error decreasing, policy entropy not collapsed.
4. `python -m muzero.main --config configs/bitfall.yaml --mode play --checkpoint <latest>` — pygame window opens, BitFall board animates, agent plays visibly sensible moves, HUD shows accumulated score rising.
5. `python -m muzero.main --config configs/tictactoe.yaml --mode train` then `--mode play` — same stack trains TTT with zero source edits (only YAML swap). Watch the agent never lose to a random opponent and draw against itself. This is the generality proof.
6. `python -m muzero.main --config configs/bitfall.yaml --mode demo --checkpoint <latest>` — produces a short animated recording of one u-MCTS search growing. This is the video's u-MCTS segment.
