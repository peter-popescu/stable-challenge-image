#!/bin/bash
# Five one-setting comparisons against the existing AdamW holdout baseline.
# Examples: bash submit_dino_followups.sh --constraint=l40s
# SEED=123 EXPERIMENT=global-crop bash submit_dino_followups.sh --constraint=l40s
# EXPERIMENT=baseline SPLIT_DIR=data/holdout-confirm bash submit_dino_followups.sh
# DRY_RUN=1 prints commands without submitting jobs.
set -euo pipefail
seed="${SEED:-42}"
split="${SPLIT_DIR:-data/holdout-v1}"
experiment="${EXPERIMENT:-all}"
case "$experiment" in
    all|baseline|global-crop|no-local|less-blur|koleo001|koleo005) ;;
    *) echo "Unknown EXPERIMENT: $experiment" >&2; exit 2 ;;
esac
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a nonnegative integer" >&2; exit 2
fi
for file in train.parquet holdout.parquet split.json; do
    if [[ ! -f "$split/$file" ]]; then
        echo "Missing $split/$file. Reuse the fixed split, or prepare a confirmation split first." >&2
        exit 2
    fi
done
for variant in baseline global-crop no-local less-blur koleo001 koleo005; do
    if [[ "$experiment" == all && "$variant" == baseline ]]; then continue; fi
    if [[ "$experiment" != all && "$experiment" != "$variant" ]]; then continue; fi
    global_min=0.4; crops=2; sigma=2.0; koleo=0
    case "$variant" in
        global-crop) global_min=0.6 ;;
        no-local) crops=0 ;;
        less-blur) sigma=0.8 ;;
        koleo001) koleo=0.01 ;;
        koleo005) koleo=0.05 ;;
    esac
    command=(sbatch --partition=gpu --gres=gpu:1 --time=03:00:00 "$@" dino.sbatch
        --backbone resnet18_wide --epochs 500 --schedule-epochs 500
        --optimizer adamw --lr 0.0015 --weight-decay 0.01
        --batch-size 64 --local-crops "$crops" --seed "$seed" --teacher-momentum 0.996
        --global-crop-min "$global_min" --blur-sigma-max "$sigma" --koleo-weight "$koleo"
        --local-crop-min 0.05 --local-crop-max 0.4
        --color-jitter-strength 1 --solarize-probability 0.2 --rotate90-probability 0
        --data "$split/train.parquet" --probe-data "$split/holdout.parquet"
        --probe-support-data "$split/train.parquet"
        --probe-every 5 --probe-low-shot 5 --probe-repeats 10 --probe-seed 2026
        --wandb-name "dino-${split##*/}-${variant}-500ep-seed${seed}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done
