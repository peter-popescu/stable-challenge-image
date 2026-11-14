#!/bin/bash
# Optional nine-run LR x teacher-momentum search. Run from the project directory.
# Extra arguments are Slurm options, e.g. bash submit_dino_grid.sh --constraint=l40s
set -euo pipefail

for lr in 0.0007 0.001 0.0015; do
    for momentum in 0.996 0.999 0.9995; do
        sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 "$@" dino.sbatch \
            --backbone resnet18_wide \
            --epochs 500 --schedule-epochs 500 \
            --batch-size 64 --local-crops 2 --seed 42 \
            --lr "$lr" --weight-decay 0.01 --teacher-momentum "$momentum" \
            --wandb-name "wide-500ep-lr${lr}-ema${momentum}-wd0.01-seed42"
    done
done
