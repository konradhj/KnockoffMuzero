# KnockoffMuZero

A from-scratch MuZero knockoff for IT-3105 (AI Programming, Spring 2025). Implements u-MCTS over learned abstract states with three interlinked neural networks (representation, dynamics, prediction), trained on-policy via BPTT in JAX + Equinox.

Two SimWorlds plug into the same AI core to demonstrate generality:
- **BitFall** — a scrolling-debris arcade game (1-player, continuous, no terminal).
- **TicTacToe** — a 2-player, terminal board game.

Swapping between them is a config-file change only. No source edits.

## Layout

```
configs/          YAML configs (single source of truth per run)
muzero/
  simworlds/      GAME-SPECIFIC (base + bitfall + tictactoe)
  ai/             GAME-AGNOSTIC (nothing under here imports from simworlds/)
    nn/           TriNet, NeuralNetworkManager, BPTT loss
    search/       u-MCTS, UMCTSNode/Edge, AbstractStateManager
    rl/           EpisodeBuffer
    types.py
  rl/rlm.py       The ONE mediator holding both SimWorld and NNM
  viz/            Pygame renderer, matplotlib dashboard, tree viewer
  io/             Checkpoint + structured logging
  main.py         CLI: train | play | demo
tests/            19 tests (incl. import-boundary + end-to-end overfit)
scripts/          audit_config.py
```

The AI core never imports from `simworlds/`. This is enforced by `tests/test_import_boundary.py`.

## Setup

```bash
conda activate it3105-jax           # or your preferred Python 3.10+ env
pip install jax jaxlib equinox optax numpy pyyaml pygame matplotlib pytest
```

## Running

Training (BitFall):
```bash
JAX_PLATFORMS=cpu python -m muzero.main --config configs/bitfall.yaml
```

Training (TicTacToe) — same stack, YAML swap:
```bash
JAX_PLATFORMS=cpu python -m muzero.main --config configs/tictactoe.yaml
```

Play with a trained checkpoint (opens pygame window, actor mode — NN_r -> NN_p, no tree search):
```bash
JAX_PLATFORMS=cpu python -m muzero.main \
    --config configs/bitfall.yaml \
    --mode play \
    --checkpoint checkpoints/bitfall_run_001/final.eqx
```

Demo one u-MCTS search from a fresh state (for the video):
```bash
JAX_PLATFORMS=cpu python -m muzero.main \
    --config configs/bitfall.yaml \
    --mode demo \
    --checkpoint checkpoints/bitfall_run_001/final.eqx
```

Outputs go to `logs/<run_name>/` (training_dashboard.png, run.jsonl, demo_tree.png) and `checkpoints/<run_name>/` (.eqx files).

## Evaluation / baselines

After training, compare the learned agent against random and always-stay baselines:

```bash
JAX_PLATFORMS=cpu python scripts/evaluate.py \
    --config configs/bitfall.yaml \
    --checkpoint checkpoints/bitfall/final.eqx \
    --episodes 100
```

Reports mean/std/min/max return per policy (`random`, `stay`, `actor` = NN_r+NN_p greedy, `mcts` = full tree search). Use this table in the video.

## Running on IDUN (NTNU HPC cluster)

One-time setup on an IDUN login node:

```bash
ssh konradj@idun.hpc.ntnu.no
cd /cluster/home/konradj   # or wherever you clone the repo
git clone <this repo>       # or rsync from your laptop
cd KnockoffMuzero
bash scripts/idun/setup_env.sh     # creates conda env 'it3105-jax' once
```

Submit a training job:

```bash
# Default: trains configs/bitfall_big.yaml on a GPU node (2 h wall-time).
sbatch scripts/idun/train.slurm

# Override the config via env var:
sbatch --export=CONFIG=configs/bitfall.yaml scripts/idun/train.slurm

# CPU-only variant (often faster for small nets because MCTS is the bottleneck):
sbatch --export=CONFIG=configs/bitfall.yaml scripts/idun/train_cpu.slurm

# Array job across a sweep of configs (drop configs into configs/sweep/ first):
N=$(ls configs/sweep/*.yaml | wc -l)
sbatch --array=0-$((N-1)) scripts/idun/sweep.slurm

# Evaluate a checkpoint:
sbatch --export=CONFIG=configs/bitfall.yaml,CKPT=checkpoints/bitfall/final.eqx,EPISODES=200 \
       scripts/idun/evaluate.slurm

# Queue inspection:
squeue -u konradj
scancel <JOBID>
```

Slurm stdout/stderr go to `logs/slurm/<job>_<id>.out|.err`. Training artifacts (checkpoints + `run.jsonl` + `training_dashboard.png`) go to `checkpoints/<run>/` and `logs/<run>/` as usual.

**Note on GPU speedup**: u-MCTS runs in pure Python (small M_s per move, tight loop with many small NN forward passes). GPU helps the BPTT training step by 5–10×, but MCTS itself dominates wall-clock time. Use the CPU queue unless you're training a big network or large batch.

## Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests/
```

Runs in under 10s. Includes:
- `test_import_boundary.py` — the critical-divide guarantee
- `test_end_to_end_overfit.py` — loss-drops canary on a single synthetic episode
- shape, invariant, and game-rule checks

## Config file

Every pivotal parameter lives in the YAML. There are no literals to hunt for in the source. Run `python scripts/audit_config.py` for a heuristic sweep.

Sections: `run`, `game`, `umcts`, `nn`, `training`, `logging`, `viz`. See [configs/bitfall.yaml](configs/bitfall.yaml) for the full schema.
