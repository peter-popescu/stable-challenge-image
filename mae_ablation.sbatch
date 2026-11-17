#!/bin/bash
#SBATCH --job-name=mae-abl
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Step 4 ablations, one after another on the same GPU, then one score table at the end.
#   sbatch mae_ablation.sbatch
# Runs that already finished (model.onnx exists) are skipped, and an unfinished run resumes from its
# last.pt, so if the job hits the time limit just submit it again.

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the project directory}"

export OMP_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

RUNS=(
  "dec1   --dec-depth 1"
  "dec8   --dec-depth 8"
  "mask60 --mask-ratio 0.6"
  "mask90 --mask-ratio 0.9"
)

for run in "${RUNS[@]}"; do
  read -r name args <<< "$run"
  out="runs/mae-$name"
  if [[ -f "$out/model.onnx" ]]; then
    echo "=== $name: already finished, skipping"
    continue
  fi
  echo "=== $name: mae_train.py $args  ($(date))"
  # shellcheck disable=SC2086
  .venv/bin/python mae_train.py --workers 6 $args --out "$out" || echo "=== $name FAILED, continuing with the next run"
done

echo "=== scores ($(date))"
models=(runs/mae-baseline/model.onnx)
for run in "${RUNS[@]}"; do
  read -r name _ <<< "$run"
  [[ -f "runs/mae-$name/model.onnx" ]] && models+=("runs/mae-$name/model.onnx")
done
.venv/bin/python challenge_score.py "${models[@]}" | tee runs/step4_scores.txt

echo "Finished $(date)"
