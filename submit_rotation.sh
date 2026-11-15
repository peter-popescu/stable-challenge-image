#!/bin/bash
# One rotation-only ablation vs the existing seed-42 holdout baseline.
set -euo pipefail
if [[ ! -f data/holdout-v1/train.parquet || ! -f data/holdout-v1/holdout.parquet ]]; then
    echo "Reuse the existing data/holdout-v1 split from the baseline experiment." >&2
    exit 2
fi
sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 "$@" dino.sbatch \
    --backbone resnet18_wide --epochs 500 --schedule-epochs 500 \
    --batch-size 64 --local-crops 2 --seed 42 \
    --lr 0.0015 --weight-decay 0.01 --teacher-momentum 0.996 \
    --local-crop-min 0.05 --local-crop-max 0.4 \
    --color-jitter-strength 1.0 --solarize-probability 0.2 \
    --rotate90-probability 0.5 \
    --data data/holdout-v1/train.parquet \
    --probe-data data/holdout-v1/holdout.parquet \
    --probe-support-data data/holdout-v1/train.parquet \
    --wandb-name "aug-holdout-rotate90-p0.5-500ep-seed42"
