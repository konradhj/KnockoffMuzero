# KnockoffMuZero — Experiments Log

A running list of the training tweaks we tried, what each one was meant to fix, and how the numbers moved. Useful source material for the video.

Baseline metrics we track each run:
- **loss_pi** — policy cross-entropy against u-MCTS visit distribution (sum over w+1 unrolls). Uniform over 3 actions × 5 unrolls ≈ 5.5.
- **loss_v** — value MSE (bootstrapped n-step return target).
- **loss_r** — reward MSE at each unroll step.
- **entropy** — mean root-policy Shannon entropy over an episode. log 3 ≈ 1.098 = fully uniform.
- **return** — sum of scaled rewards per episode (BitFall `reward_scale=0.25` so scaled return × 4 ≈ raw return).

---

## Run 001 — baseline, raw rewards, M_s=40

Config: 6×6 grid, 3 receptors, rollout enabled, reward_scale implicit 1.0, M_s=40.

Result after 200 eps (~12.5 min on CPU):
- loss_v ≈ 60, loss_r ≈ 0.3, loss_pi ≈ 5.4 (flat)
- mean return ≈ −35
- entropy ≈ 1.03

**Diagnosis**: value targets massive (raw rewards ±6 compounded over n_step=8). Loss is dominated by value MSE; policy never differentiates.

---

## Run 002 — reward scaling, disable rollout, smaller grid, higher M_s

Techniques added:
1. **Reward scaling** (`reward_scale=0.25`): rewards and thus value targets shrink to ~[-1, 1]. Standard MuZero paper trick — they use categorical value heads for the same reason; scalar MSE just needs the inputs tamed.
2. **Disable in-tree rollout** (`rollout_enabled=false`): with stochastic dynamics (random new debris row each step), NN_d/NN_p rollouts inject variance. Use `NN_p.value(leaf_sigma)` directly as the bootstrap.
3. **Smaller game** (4×4 grid, 1 receptor, horizon 40): per PDF's guidance "start tiny, scale up." Reduces state-space + signal-to-noise ratio issues for the policy.
4. **Higher M_s** (40 → 80) and **higher c_ucb** (1.25 → 1.5): more simulations + more exploration to fight prior collapse.
5. **Visit-count temperature annealing** on the action sampler (T: 1.0 → 0.25 across training): more exploration early, exploitation late. This only affects the behavior policy — training targets are still raw visit counts.

Result after 150 eps (~5.5 min on CPU):
- loss_v ≈ 1.0 (60× drop ✓)
- loss_r ≈ 0.02 (15× drop ✓)
- loss_pi ≈ 5.3 (still stuck)
- mean return ≈ −3 (scaled; ≈ −12 unscaled)
- entropy ≈ 1.07 (basically uniform)

**Diagnosis**: value and reward networks learn cleanly. Policy still uniform because the u-MCTS visit distribution itself is near-uniform: with uniform NN_p priors and small Q differences, UCB's exploration term equalizes N across actions. loss_pi of 5.3 = log(3) × ~5 unrolls = 5.5 → essentially uniform targets, so NN_p has nothing non-uniform to learn from.

---

## Run 003 — policy target temperature T=0.5

Tweak: sharpened the **stored policy target** with T=0.5 (visits^2 / sum).

Result after 150 eps:
- loss_pi ≈ 5.3 (unchanged)
- mean return ≈ −3.0 (unchanged)
- entropy ≈ 1.07 (unchanged)

**Diagnosis**: no-op. Sharpening near-uniform visits still yields near-uniform. The problem is upstream: u-MCTS isn't producing differentiated visit counts in the first place. Confirmed theory: UCB + uniform priors + small Q differences → round-robin expansion.

Code change: added `policy_target_temperature` + `q_policy_mix` + `q_policy_temperature` to `TrainingConfig`. Also exposed per-action Q values on `SearchResult`. Added per-episode diagnostics to the log: `visit_spread` (max−min visit fraction) and `q_spread` (max−min root Q).

---

## Run 004 — lower c_ucb (0.8)

Tweak: cut the UCB exploration constant from 1.5 → 0.8 to let Q dominate visit selection once any signal appears. (Kept as config file but not run — superseded by run_005.)

---

## Run 005 — Q-mixed policy target + bigger tree

Tweaks:
- **Q-mixed policy target** (`q_policy_mix=0.4`, `q_policy_temperature=0.25`): 60 % from sharpened visits, 40 % from softmax(Q/0.25). If the value head has learned any action preference, the target will reflect it even when visits are uniform.
- **Bigger tree**: M_s 80 → 160, d_max 6 → 8. More simulations increase chance of visit differentiation.
- **Lower c_ucb** (1.5 → 1.0) to let Q values affect selection once present.

---

## Diagnostic: probe_representation.py (after run 005)

Ran `scripts/probe_representation.py` on 5 very different BitFall initial states:

```
sigma_mean_pairwise_dist: 0.286   (in a 64-d vector bounded to [0,1])
sigma_std_per_dim (mean):  0.022  (!!)
sigma_range:               0.0 to 1.0
Q across 3 actions:        [-1.14, -1.11, -1.12]   (spread = 0.03)
NN_p direct policy:        [0.347, 0.345, 0.308]   (uniform)
```

**Smoking gun**: σ only varies by std 0.022 per dimension across radically different game states. The representation is nearly collapsed. With σ indistinguishable across states, NN_p and NN_d cannot possibly produce differentiated outputs — no amount of MCTS or policy-target tricks can rescue this.

Culprit: the per-sample min-max normalization `(x - min(x)) / (max(x) - min(x))` I inherited from MuZero-paper-style implementations. Applied PER-SAMPLE, it strips out all meaningful cross-sample variance: every σ ends up spanning [0,1] by construction, regardless of input.

## Run 006 — replace min-max with tanh bounding

Fix: `σ = tanh(MLP(…))` in both NN_r and NN_d. Keeps σ bounded without normalizing away cross-sample variance. Everything else identical to run 002 (same 4×4 / 1 receptor / M_s=80).

Per-30-episode summary:

| episodes | mean return | entropy | visit_spread | q_spread |
|---:|---:|---:|---:|---:|
| 0–29   | −2.88 | 1.07 | 0.17 | 0.03 |
| 30–59  | −2.67 | 1.06 | 0.20 | 0.04 |
| 60–89  | −2.72 | 1.06 | 0.21 | 0.03 |
| 90–119 | −2.49 | 1.04 | 0.25 | 0.04 |
| **120–149** | **+0.75** | **0.70** | **0.58** | **0.38** |

The learning kicks in around episode 100 and accelerates sharply in the final 30. Return goes from consistently negative to positive; policy entropy drops from near-uniform (log 3 = 1.098) down to 0.70; Q values finally differentiate by an order of magnitude.

Probe of the final checkpoint:
```text
sigma_range:         -0.18 to 0.25           (bounded via tanh, not saturated)
Q across 3 actions:   [-0.70, -0.39, -0.58]   spread = 0.31
NN_p direct policy:   [0.23, 0.51, 0.26]      clearly non-uniform
```

**Takeaway**: the per-sample min-max normalization was fighting the representation. Replacing with tanh (a simple bounded nonlinearity that preserves cross-sample variance) was THE change that unlocked learning. Everything else — reward scaling, disabled rollout, policy target temperature, Q-mix — was plumbing.

---

## Run 007 — 300 episodes to confirm sustained learning

Same config as run 006 but doubled episode count.

Per-50-episode summary:

| episodes | mean return | entropy | visit_spread | q_spread |
|:---:|:---:|:---:|:---:|:---:|
| 0–49    | −2.80 | 1.07 | 0.18 | 0.03 |
| 50–99   | −2.58 | 1.05 | 0.22 | 0.03 |
| 100–149 | −1.36 | 0.87 | 0.42 | 0.17 |
| 150–199 | **+2.47** | 0.18 | 0.92 | 1.33 |
| 200–249 | **+2.76** | 0.05 | 0.99 | 1.71 |
| 250–299 | **+2.64** | 0.05 | 0.99 | 1.82 |

Final losses: loss_pi ≈ 1.0 (down from 5.3), loss_v ≈ 2.0, loss_r ≈ 0.05.

Probe of final checkpoint:
```text
sigma_std_per_dim (mean): 0.062   (3× what we had in run 005 — representation now varies)
sigma_range:              -0.80 to 0.70   (uses most of tanh range, not saturated)
Q across 3 actions:        [-1.22, +0.91, -0.68]   spread = 2.13
NN_p direct policy:        [0.02, 0.94, 0.03]      near-deterministic on "stay"
NN_p direct value:         +0.64
```

**Sustained learning confirmed.** Agent converged to a near-deterministic "stay" strategy, which is reasonable for a length-1 receptor on a 4×4 grid: moving costs; positioning a 1-wide receptor in a fixed column catches whatever debris happens to fall on it. The policy head is no longer a uniform distribution — it's a confident decision, and the Q values back it up.

This config is now promoted to `configs/bitfall.yaml`.

---

## What to emphasize in the 10-minute video

The debugging arc makes a clean narrative:

1. **Naive MuZero (run 001) collapses**: loss_v = 60, policy stuck at uniform. Problem: value targets are huge because BitFall rewards are unscaled.
2. **Reward scaling + disable rollout (run 002)** fixes value/reward losses (60× and 15× reductions) but policy stays uniform. Problem is now upstream.
3. **Sharpening the stored policy target (run 003)** is a no-op. Sharpening uniform is still uniform.
4. **Bigger tree + Q-mix (run 005)** is also a no-op. Show the q_spread plot: 0.03 across the whole run — Q values barely differentiate across actions.
5. **Diagnostic probe reveals representation collapse**: σ_std = 0.022 across radically different states. Smoking gun for per-sample min-max normalization.
6. **Replace min-max with tanh (run 006)**: learning kicks in around episode 100, return flips positive by episode 120.
7. **Confirmed over 300 episodes (run 007)**: agent converges on a near-deterministic strategy, Q spread grows 60× from 0.03 to 1.8, return stabilizes at +2.7.

The lesson: MuZero's bundled-in tricks (σ normalization, rollouts, categorical value heads) are not all universally helpful. When a trick actively hurts, it's usually because it was designed for a different regime (Atari-scale networks, large-batch training, deep unrolls).

---

## Lessons to emphasize in the video

- **Reward scale matters more than you'd think.** A 60× reduction in loss_v just from dividing rewards by a constant.
- **Stochastic games fight rollout.** In-tree rollout sampled via NN_p/NN_d adds noise more than signal; disabling it recovered a clean value signal.
- **u-MCTS visit counts do not automatically differentiate.** UCB's exploration term plus initial uniform NN_p priors equalize visit counts unless something external breaks the symmetry (larger M_s, smaller c_ucb, temperature-sharpened targets, or a value head that's already useful).
- **Start tiny.** Going from 6×6 with 3 receptors down to 4×4 with 1 receptor made debugging possible at all.
