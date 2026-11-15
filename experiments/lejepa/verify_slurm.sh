#!/usr/bin/env bash
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
: "${SLURM_JOB_ID:?Run within Slurm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export SPT_CACHE_DIR="$PWD/experiment_outputs/cache/stable-pretraining"
export HF_HOME="$PWD/experiment_outputs/cache/huggingface"
export XDG_CACHE_HOME="$PWD/experiment_outputs/cache/xdg"
export MPLCONFIGDIR="$PWD/experiment_outputs/cache/matplotlib"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON="$PWD/.venv/bin/python"
OUT="$PWD/experiment_outputs/verification"
if [[ "${SKIP_UNIT_TESTS:-0}" != "1" ]]; then
    "$PYTHON" -m pytest tests --basetemp="$OUT/pytest" -q
fi
"$PYTHON" -m experiments.lejepa.main --smoke --accelerator cpu --data data/sample.parquet --output "$OUT/smoke"
"$PYTHON" -m experiments.lejepa.export "$OUT/smoke/checkpoints/last.ckpt" "$OUT/smoke/model.onnx"
"$PYTHON" test_submission.py "$OUT/smoke/model.onnx"
"$PYTHON" evaluate.py "$OUT/smoke/model.onnx" --data data/sample.parquet --out "$OUT/sample_probe.json"
"$PYTHON" -m experiments.lejepa.diagnostics "$OUT/smoke/checkpoints/last.ckpt" --data data/sample.parquet --output "$OUT/plots"
"$PYTHON" -m experiments.lejepa.diagnostics "$OUT/smoke/checkpoints/last.ckpt" --data data/sample.parquet --output "$OUT/shared" --shared
"$PYTHON" -m experiments.lejepa.main --smoke --accelerator cpu --data data/sample.parquet --output "$OUT/smoke" --resume "$OUT/smoke/checkpoints/last.ckpt"
