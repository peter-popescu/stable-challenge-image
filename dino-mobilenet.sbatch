#!/bin/bash
#SBATCH --job-name=dino-mobilenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=dino-mobilenet-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if (( ${SLURM_CPUS_PER_TASK:-0} < 6 )); then
    echo "Request six CPU cores for four loader workers and two training threads." >&2
    exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
    echo "Run uv sync --locked --python 3.12 first." >&2
    exit 2
fi

srun .venv/bin/python main.py \
    --backbone mobilenet_v3_large \
    --accelerator gpu \
    --precision 16-mixed \
    --epochs 300 \
    --batch-size 64 \
    --lr 0.0005 \
    --seed 42 \
    --workers 4 \
    --probe-every 5 \
    --wandb \
    --wandb-project dino-experiment \
    --wandb-name "mobilenet-dino-${SLURM_JOB_ID}" \
    --output "runs/dino-mobilenet-${SLURM_JOB_ID}" \
    "$@"
