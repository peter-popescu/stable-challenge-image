#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run this sweep through Slurm}"
: "${EXPERIMENT_STORAGE:?Set repository-local experiment storage}"
export UV_CACHE_DIR="$EXPERIMENT_STORAGE/cache/uv"
export SPT_CACHE_DIR="$EXPERIMENT_STORAGE/cache/stable-pretraining"
export HF_HOME="$EXPERIMENT_STORAGE/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_STORAGE/cache/xdg"
export MPLCONFIGDIR="$EXPERIMENT_STORAGE/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv/bin/python -m pytest tests/test_sweep.py tests/test_cross_validate.py tests/test_lejepa.py tests/test_extend.py tests/test_seeds.py \
    --basetemp="$EXPERIMENT_STORAGE/sweep-preflight-$SLURM_JOB_ID" -q
sweep_module="${LEJEPA_SWEEP_MODULE:-experiments.lejepa.sweep}"
exec .venv/bin/python -u -m "$sweep_module" "$@"
