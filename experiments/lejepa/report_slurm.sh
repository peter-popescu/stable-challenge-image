#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export SPT_CACHE_DIR="$PWD/experiment_outputs/cache/stable-pretraining"
export HF_HOME="$PWD/experiment_outputs/cache/huggingface"
export XDG_CACHE_HOME="$PWD/experiment_outputs/cache/xdg"
export MPLCONFIGDIR="$PWD/experiment_outputs/cache/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
analysis_module="${LEJEPA_ANALYSIS_MODULE:-experiments.lejepa.report}"
if [[ "$analysis_module" == "experiments.lejepa.error_analysis" ]]; then
    .venv/bin/python -m pytest tests/test_error_analysis.py \
        --basetemp="$PWD/experiment_outputs/error-analysis-tests-$SLURM_JOB_ID" -q
fi
if [[ "$analysis_module" == "experiments.lejepa.cross_validate" ]]; then
    .venv/bin/python -m pytest tests/test_cross_validate.py \
        --basetemp="$PWD/experiment_outputs/cross-validation-tests-$SLURM_JOB_ID" -q
fi
if [[ "$analysis_module" == "experiments.lejepa.milestones" ]]; then
    .venv/bin/python -m pytest tests/test_milestones.py tests/test_cross_validate.py \
        --basetemp="$PWD/experiment_outputs/milestone-tests-$SLURM_JOB_ID" -q
fi
exec .venv/bin/python -u -m "$analysis_module" "$@"
