#!/bin/bash
#SBATCH --job-name=dino
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=dino-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"

# Four data-loader processes plus two threads for the main training process.
workers=4
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if (( ${SLURM_CPUS_PER_TASK:-0} < workers + OMP_NUM_THREADS )); then
    echo "Request at least six CPU cores for four workers and two training threads." >&2
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Run uv sync --locked --python 3.12 in the project directory first." >&2
    exit 2
fi

echo "DINO: accelerator=gpu tasks=1 cores=$SLURM_CPUS_PER_TASK workers=$workers training_threads=$OMP_NUM_THREADS"
srun .venv/bin/python main.py \
    --accelerator gpu \
    --precision 16-mixed \
    --epochs 100 \
    --batch-size 64 \
    --workers "$workers" \
    --wandb \
    --wandb-project dino-experiment \
    --wandb-name "oscar-${SLURM_JOB_ID}" \
    --output "runs/dino-${SLURM_JOB_ID}" \
    "$@"
