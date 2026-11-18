#!/usr/bin/env bash
# Default: local-min10 and jitter-strong sequentially; optional specs are explicit.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm after matrix review}"
export EXPERIMENT_STORAGE="$PWD/experiment_outputs"
export UV_CACHE_DIR="$EXPERIMENT_STORAGE/cache/uv"
export SPT_CACHE_DIR="$EXPERIMENT_STORAGE/cache/stable-pretraining"
export HF_HOME="$EXPERIMENT_STORAGE/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_STORAGE/cache/xdg"
export MPLCONFIGDIR="$EXPERIMENT_STORAGE/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv/bin/python -m pytest tests/test_augmentation_study.py tests/test_lejepa.py \
    tests/test_cross_validate.py tests/test_sweep.py \
    --basetemp="$EXPERIMENT_STORAGE/augmentation-preflight-$SLURM_JOB_ID" -q
exec .venv/bin/python -u -m experiments.lejepa.augmentation_study "$@"
