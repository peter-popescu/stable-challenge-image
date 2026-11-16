#!/bin/bash
# CPU-only comparison of the original full-data baseline and KoLeo model.
# Optional arguments replace the job IDs: sbatch compare_probes.sbatch 9061570 9188109
#SBATCH --job-name=probe-compare
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=probe-compare-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing project .venv/bin/python" >&2
    exit 2
fi
if (( $# == 0 )); then
    set -- 9061570 9188109
fi
for model_job in "$@"; do
    if [[ ! "$model_job" =~ ^[0-9]+$ ]]; then
        echo "Expected numeric training job IDs" >&2
        exit 2
    fi
    model="runs/dino-resnet-${model_job}/model.onnx"
    cache_dir="runs/diagnostics-${model_job}"
    if [[ ! -f "$model" ]]; then
        echo "Missing $model" >&2
        exit 2
    fi
    if [[ ! -f "$cache_dir/embeddings.npz" ]]; then
        srun .venv/bin/python diagnose.py "$model" --data data/train.parquet \
            --threads "$OMP_NUM_THREADS" --save-embeddings --out "$cache_dir"
    fi
    srun .venv/bin/python compare_probes.py "$cache_dir/embeddings.npz" \
        --model "$model" --folds 5 --inner-folds 3 --seed 2026 \
        --out "runs/probe-comparison-${model_job}-${SLURM_JOB_ID}"
done
