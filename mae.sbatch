#!/bin/bash
#SBATCH --job-name=mae
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="titan_rtx|rtx_a6000|l40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out

# Usage:  sbatch mae.sbatch --out runs/mae-baseline [any other mae_train.py options]
# Re-submitting with the same --out resumes from runs/.../last.pt.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project directory}"

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Assigned GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv

.venv/bin/python mae_train.py --workers 6 "$@"

echo "Finished $(date)"
