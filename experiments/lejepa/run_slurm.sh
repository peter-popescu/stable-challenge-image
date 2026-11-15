#!/usr/bin/env bash
# Submit with site-approved resource flags on the sbatch command line.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through sbatch or srun}"
: "${EXPERIMENT_STORAGE:?Set an approved storage directory inside the repository (or explicitly authorize another path)}"
: "${UV_BIN:?Set the absolute path to uv}"
export UV_CACHE_DIR="$EXPERIMENT_STORAGE/cache/uv"
export SPT_CACHE_DIR="$EXPERIMENT_STORAGE/cache/stable-pretraining"
export HF_HOME="$EXPERIMENT_STORAGE/cache/huggingface"
export XDG_CACHE_HOME="$EXPERIMENT_STORAGE/cache/xdg"
export MPLCONFIGDIR="$EXPERIMENT_STORAGE/cache/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ "${LEJEPA_PREFLIGHT:-0}" == "1" ]]; then
    "$UV_BIN" run --frozen --offline --no-sync python -m pytest tests/test_lejepa.py \
        --basetemp="$EXPERIMENT_STORAGE/preflight-$SLURM_JOB_ID" -q
fi
exec "$UV_BIN" run --frozen --offline --no-sync python -m experiments.lejepa.main "$@"
