#!/bin/bash
# Run ONCE on an IDUN login node to create the conda env this project uses.
# After that, every SLURM job just activates it — no reinstall.
#
#   ssh konradj@idun.hpc.ntnu.no
#   cd /cluster/home/konradj/KnockoffMuzero
#   bash scripts/idun/setup_env.sh
#
# For a CPU-only install (fine: this project's bottleneck is MCTS on the CPU,
# and the NNs are small), drop `-cuda12_pip` from the jax install line.

set -euo pipefail

ENV_NAME="it3105-jax"

module purge
module load Anaconda3/2023.09-0

# Create the env if missing.
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
    conda create -y -n "${ENV_NAME}" python=3.11
fi

# Activate for the duration of this script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install --upgrade pip
# Install JAX with CUDA support. On IDUN GPU nodes this auto-picks the right wheel.
# For CPU-only runs, swap the first line for:  pip install "jax[cpu]"
pip install --upgrade "jax[cuda12]" "jaxlib"
pip install equinox optax numpy pyyaml matplotlib pytest

# pygame is only needed for the play/demo modes (not training). Skip on headless
# nodes unless you plan to run --mode play remotely.
pip install pygame || true

python - <<'PY'
import jax, equinox, optax
print("jax:", jax.__version__, "devices:", jax.devices())
print("equinox:", equinox.__version__, "optax:", optax.__version__)
PY

echo "Environment '${ENV_NAME}' ready. Activate later with:"
echo "  module load Anaconda3/2023.09-0 && conda activate ${ENV_NAME}"
