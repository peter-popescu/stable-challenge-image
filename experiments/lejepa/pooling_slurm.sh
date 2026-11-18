#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export SPT_CACHE_DIR="$PWD/experiment_outputs/cache/stable-pretraining"
export HF_HOME="$PWD/experiment_outputs/cache/huggingface"
export XDG_CACHE_HOME="$PWD/experiment_outputs/cache/xdg"
export MPLCONFIGDIR="$PWD/experiment_outputs/cache/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -m pytest tests/test_architecture_pooling.py tests/test_cross_validate.py \
 --basetemp="$PWD/experiment_outputs/backbone-pooling/frozen-readouts/tests-$SLURM_JOB_ID" -q
exec .venv/bin/python -u -m experiments.lejepa.pooling_study "$@"
