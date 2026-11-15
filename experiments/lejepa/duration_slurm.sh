#!/usr/bin/env bash
# One reproducible 400-epoch tuned run, followed by milestone probe evaluations.
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export SPT_CACHE_DIR="$PWD/experiment_outputs/cache/stable-pretraining"
export HF_HOME="$PWD/experiment_outputs/cache/huggingface"
export XDG_CACHE_HOME="$PWD/experiment_outputs/cache/xdg"
export MPLCONFIGDIR="$PWD/experiment_outputs/cache/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export EXPERIMENT_STORAGE="$PWD/experiment_outputs"
export LEJEPA_DURATION_OUTPUT="${LEJEPA_DURATION_OUTPUT:-$PWD/experiment_outputs/duration/tuned-seed43-400}"
DURATION_CONFIG="$PWD/experiments/lejepa/config_400_seed43.json"
DURATION_DATA="$PWD/data/train.parquet"
DURATION_FOLDS="$PWD/experiment_outputs/validation/stratified-5fold-seed42.json"
DURATION_PYTHON="$PWD/.venv/bin/python"
"$DURATION_PYTHON" -m pytest tests/test_lejepa.py tests/test_cross_validate.py \
    --basetemp="$EXPERIMENT_STORAGE/duration-preflight-$SLURM_JOB_ID" -q
train_args=(--config "$DURATION_CONFIG" --data "$DURATION_DATA" \
            --output "$LEJEPA_DURATION_OUTPUT" --accelerator gpu --retain-every 50)
if [[ -f "$LEJEPA_DURATION_OUTPUT/checkpoints/last.ckpt" ]]; then
    train_args+=(--resume "$LEJEPA_DURATION_OUTPUT/checkpoints/last.ckpt")
fi
"$DURATION_PYTHON" -u -m experiments.lejepa.main "${train_args[@]}"
for duration_epoch in 200 300 400; do
    if [[ "$duration_epoch" == 400 ]]; then
        duration_checkpoint="$LEJEPA_DURATION_OUTPUT/checkpoints/last.ckpt"
        duration_destination="$LEJEPA_DURATION_OUTPUT"
    else
        duration_saved_epoch=$(printf '%03d' "$((duration_epoch - 1))")
        duration_matches=("$LEJEPA_DURATION_OUTPUT/checkpoints/retained-epoch-$duration_saved_epoch-step-"*.ckpt)
        if [[ "${#duration_matches[@]}" != 1 || ! -f "${duration_matches[0]}" ]]; then
            echo "Missing or ambiguous checkpoint at epoch $duration_epoch" >&2
            exit 1
        fi
        duration_checkpoint="${duration_matches[0]}"
        duration_destination="$LEJEPA_DURATION_OUTPUT/epoch$duration_epoch"
    fi
    mkdir -p "$duration_destination"
    "$DURATION_PYTHON" -u -m experiments.lejepa.export "$duration_checkpoint" "$duration_destination/model.onnx"
    "$DURATION_PYTHON" -u -m experiments.lejepa.cross_validate \
        --model "$duration_destination/model.onnx" --data "$DURATION_DATA" \
        --folds "$DURATION_FOLDS" --output "$duration_destination/validation"
done
"$DURATION_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['LEJEPA_DURATION_OUTPUT'])
reference_path = Path('experiment_outputs/seed-repeats/baseline-vs-tuned/winner-seed-43/validation/results.json')
reference = json.loads(reference_path.read_text())
rows = [{'recipe': 'Original 200-epoch schedule', 'epoch': 200,
         **reference['results']['trained']['metrics']}]
for epoch, directory in [(200,root/'epoch200'), (300,root/'epoch300'), (400,root)]:
    result = json.loads((directory/'validation/results.json').read_text())
    for key in ['data_sha256','fold_manifest_sha256']:
        if result[key] != reference[key]:
            raise ValueError('Duration comparison data/folds changed')
    rows.append({'recipe':'400-epoch schedule', 'epoch':epoch,
                 **result['results']['trained']['metrics']})
(root/'comparison.json').write_text(json.dumps(rows,indent=2))
lines = ['# Longer training recipe, seed 43', '',
         '| Recipe | Epoch | Outer accuracy | Macro accuracy | Official local LOO |',
         '|---|---:|---:|---:|---:|']
for row in rows:
    lines.append(f'| {row["recipe"]} | {row["epoch"]} | {row["accuracy"]:.2%} | {row["macro_accuracy"]:.2%} | {row["official_local_loo_accuracy"]:.2%} |')
lines += ['', 'Seed 43 was fixed in advance rather than selected by score. The new run started from random initialization, without loading the old 200-epoch weights.',
          'LR 0.003, SIGReg 0.005, batch 128, seed, augmentation and warmup are unchanged. Only the planned cosine horizon changes from 200 to 400.',
          'This compares duration and the associated LR trajectory, not only an additional number of optimizer steps. Scores are fixed-fold development results, not unbiased held-out performance.',
          'The milestone probes were run after training; no validation labels contributed to encoder gradients. No leaderboard submission was made.']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines), flush=True)
PY
