#!/bin/sh
# Evaluation job: runs the four baseline policies and writes a table.
#
#   sbatch --export=CONFIG=configs/bitfall_big.yaml,CKPT=checkpoints/bitfall_big/final.eqx,EPISODES=200 \
#          scripts/idun/evaluate.slurm
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=muzero-eval
#SBATCH --time=0-00:45:00
#SBATCH --partition=CPUQ
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --mail-user=konradj@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

CONFIG="${CONFIG:?set CONFIG}"
CKPT="${CKPT:?set CKPT}"
EPISODES="${EPISODES:-100}"

mkdir -p logs/slurm
cd "${SLURM_SUBMIT_DIR}"

module purge
module load Anaconda3/2023.09-0
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate it3105-jax

export JAX_PLATFORMS=cpu
python scripts/evaluate.py --config "${CONFIG}" --checkpoint "${CKPT}" --episodes "${EPISODES}"
