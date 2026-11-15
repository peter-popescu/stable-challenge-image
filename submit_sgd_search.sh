#!/bin/bash
# Fresh SGD recipe comparison on the existing holdout split.
# These are actual peak LRs; this project does not apply automatic batch-size LR scaling.
set -euo pipefail
if [[ ! -f data/holdout-v1/train.parquet || ! -f data/holdout-v1/holdout.parquet ]]; then
    echo "Reuse the existing data/holdout-v1 split from the AdamW baseline." >&2
    exit 2
fi
for lr in 0.005 0.01 0.02; do
    sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 "$@" dino.sbatch \
        --backbone resnet18_wide --epochs 500 --schedule-epochs 500 \
        --optimizer sgd --sgd-momentum 0.9 --lr "$lr" --weight-decay 0.0001 \
        --batch-size 64 --local-crops 2 --seed 42 --teacher-momentum 0.996 \
        --local-crop-min 0.05 --local-crop-max 0.4 \
        --color-jitter-strength 1.0 --solarize-probability 0.2 --rotate90-probability 0 \
        --data data/holdout-v1/train.parquet \
        --probe-data data/holdout-v1/holdout.parquet \
        --probe-support-data data/holdout-v1/train.parquet \
        --wandb-name "sgd-holdout-lr${lr}-wd0.0001-m0.9-500ep-seed42"
done
