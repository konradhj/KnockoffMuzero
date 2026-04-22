#!/bin/sh
# SLURM array: trains several configs in parallel. One file per array index.
# Drop all configs you want run into `configs/sweep/*.yaml` first, then:
#   N=$(ls configs/sweep/*.yaml | wc -l)
#   sbatch --array=0-$((N-1)) scripts/idun/sweep.slurm
#
#SBATCH --account=share-ie-idi
#SBATCH --job-name=muzero-sweep
#SBATCH --time=0-02:00:00
#SBATCH --partition=CPUQ
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --mail-user=konradj@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

mkdir -p logs/slurm
cd "${SLURM_SUBMIT_DIR}"

mapfile -t CONFIGS < <(ls configs/sweep/*.yaml | sort)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task ${SLURM_ARRAY_TASK_ID} -> ${CONFIG}"
echo "Host: $(hostname)  Started: $(date)"

module purge
module load Anaconda3/2023.09-0
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate it3105-jax

export JAX_PLATFORMS=cpu
python -m muzero.main --config "${CONFIG}"

echo "Finished ${CONFIG}: $(date)"
