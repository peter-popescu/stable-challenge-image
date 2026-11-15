#!/bin/bash
# Matched baseline vs gentler color, seed 123, using the existing fixed image split.
# Additional arguments are Slurm options, e.g. --constraint=l40s.
set -euo pipefail
if [[ ! -f data/holdout-v1/train.parquet || ! -f data/holdout-v1/holdout.parquet ]]; then
    echo "The existing data/holdout-v1 split is required; copy/reuse the same split as seed 42." >&2
    exit 2
fi
for strength in 1.0 0.5; do
    sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 "$@" dino.sbatch \
        --backbone resnet18_wide --epochs 500 --schedule-epochs 500 \
        --batch-size 64 --local-crops 2 --seed 123 \
        --lr 0.0015 --weight-decay 0.01 --teacher-momentum 0.996 \
        --local-crop-min 0.05 --local-crop-max 0.4 --solarize-probability 0.2 \
        --color-jitter-strength "$strength" \
        --data data/holdout-v1/train.parquet \
        --probe-data data/holdout-v1/holdout.parquet \
        --probe-support-data data/holdout-v1/train.parquet \
        --wandb-name "aug-holdout-color${strength}-500ep-seed123"
done
