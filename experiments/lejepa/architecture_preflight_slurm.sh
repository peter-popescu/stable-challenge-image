#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export EXPERIMENT_STORAGE="$PWD/experiment_outputs"
export SPT_CACHE_DIR="$EXPERIMENT_STORAGE/cache/stable-pretraining"
export HF_HOME="$EXPERIMENT_STORAGE/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_STORAGE/cache/xdg"
export MPLCONFIGDIR="$EXPERIMENT_STORAGE/cache/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -m pytest tests/test_architecture_pooling.py tests/test_lejepa.py tests/test_cross_validate.py \
  --basetemp="$EXPERIMENT_STORAGE/backbone-pooling/preflight/tests-$SLURM_JOB_ID" -q
exec .venv/bin/python -u -m experiments.lejepa.architecture_preflight \
  --output "$EXPERIMENT_STORAGE/backbone-pooling/preflight"
