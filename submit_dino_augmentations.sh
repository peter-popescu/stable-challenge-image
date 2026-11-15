#!/bin/bash
# Four controlled runs. Optional --holdout must be first; remaining args are Slurm options.
# bash submit_dino_augmentations.sh --holdout --constraint=l40s
set -euo pipefail

mode=full
data_options=(--data data/train.parquet)
if [[ "${1:-}" == "--holdout" ]]; then
    shift
    mode=holdout
    if [[ ! -f data/holdout-v1/train.parquet || ! -f data/holdout-v1/holdout.parquet ]]; then
        echo "Create the fixed split first: .venv/bin/python prepare_holdout.py" >&2
        exit 2
    fi
    data_options=(--data data/holdout-v1/train.parquet
                  --probe-data data/holdout-v1/holdout.parquet
                  --probe-support-data data/holdout-v1/train.parquet)
fi

for experiment in baseline larger-local gentler-color no-solarize; do
    options=(--local-crop-min 0.05)
    case "$experiment" in
        larger-local) options=(--local-crop-min 0.15) ;;
        gentler-color) options=(--color-jitter-strength 0.5) ;;
        no-solarize) options=(--solarize-probability 0) ;;
    esac
    sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 "$@" dino.sbatch \
        --backbone resnet18_wide --epochs 500 --schedule-epochs 500 \
        --batch-size 64 --local-crops 2 --seed 42 \
        --lr 0.0015 --weight-decay 0.01 --teacher-momentum 0.996 \
        "${data_options[@]}" "${options[@]}" \
        --wandb-name "aug-${mode}-${experiment}-500ep-seed42"
done
