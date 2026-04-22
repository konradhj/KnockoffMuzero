#!/bin/sh
# CPU-only variant. Use when GPU queue is long or the game is tiny enough that
# MCTS Python overhead dominates (making GPU mostly useless).
#
#   sbatch --export=CONFIG=configs/bitfall_big.yaml scripts/idun/train_cpu.slurm
#
#SBATCH --account=ie-idi
#SBATCH --job-name=muzero-train-cpu
#SBATCH --time=0-03:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --mail-user=konradj@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

CONFIG="${CONFIG:-configs/bitfall_big.yaml}"

mkdir -p logs/slurm
cd "${SLURM_SUBMIT_DIR}"

echo "Host: $(hostname)  Config: ${CONFIG}  Started: $(date)"

module purge
module load Anaconda3/2023.09-0
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate it3105-jax

export JAX_PLATFORMS=cpu
python -m muzero.main --config "${CONFIG}"

echo "Finished: $(date)"
