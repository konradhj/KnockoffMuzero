#!/bin/sh
# Single-config training job. Launch with:
#   sbatch --export=CONFIG=configs/bitfall_big.yaml scripts/idun/train.slurm
# or edit the CONFIG line below.
#
#SBATCH --account=ie-idi
#SBATCH --job-name=muzero-train
#SBATCH --time=0-02:00:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --mail-user=konradj@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

CONFIG="${CONFIG:-configs/bitfall_big.yaml}"

mkdir -p logs/slurm
WORKDIR="${SLURM_SUBMIT_DIR}"
cd "${WORKDIR}"

echo "Host:       $(hostname)"
echo "Job:        ${SLURM_JOB_ID}  ${SLURM_JOB_NAME}"
echo "Working dir ${WORKDIR}"
echo "Config:     ${CONFIG}"
echo "Started:    $(date)"
nvidia-smi -L || true

module purge
module load Anaconda3/2023.09-0
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate it3105-jax

# XLA/JAX memory hints: let JAX grow on demand, not pre-allocate 75%.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m muzero.main --config "${CONFIG}"

echo "Finished: $(date)"
