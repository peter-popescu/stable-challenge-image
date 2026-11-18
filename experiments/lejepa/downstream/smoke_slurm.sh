#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export SPT_CACHE_DIR="$PWD/experiment_outputs/cache/stable-pretraining"
export HF_HOME="$PWD/experiment_outputs/cache/huggingface"
export MPLCONFIGDIR="$PWD/experiment_outputs/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
exec .venv/bin/python -m pytest tests/test_downstream.py tests/test_architecture_pooling.py tests/test_cross_validate.py tests/test_lejepa.py --basetemp="$PWD/experiment_outputs/green-area/preflight/tests-$SLURM_JOB_ID" -q
