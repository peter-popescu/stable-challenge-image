# DINO experiment

## Detail preservation and feature-spreading comparisons

`submit_dino_followups.sh` queues **five** fresh, one-setting comparisons with
the existing wide-ResNet AdamW baseline. Each requests one GPU, six CPU cores,
32 GiB RAM, and **three hours**. Nothing is submitted by installing this code.
Defaults remain unchanged for earlier experiments and checkpoints.

| Experiment name | Only training change from baseline |
| --- | --- |
| `global-crop` | Global minimum crop area 0.4 → 0.6 |
| `no-local` | Local crops 2 → 0; both global crops retained |
| `less-blur` | Maximum blur sigma 2.0 → 0.8 on every view; probabilities unchanged |
| `koleo001` | Add encoder KoLeo regularization, weight 0.01 |
| `koleo005` | Add encoder KoLeo regularization, weight 0.05 |

All use `resnet18_wide`, AdamW, LR 0.0015, WD 0.01, EMA 0.996,
batch 64, 500 training/schedule epochs, original color augmentation, and no
rotation. KoLeo is applied to normalized **student encoder features before the
projection head**, for the first global crop only. Neighbours are different
images in the same batch. It reuses the original student forward pass, leaving
BatchNorm updates unchanged. The loss calculation runs in float32 under mixed
precision; `train/dino_loss` and unweighted `train/koleo_loss` are logged separately
when enabled. `train/loss` includes the weighted regularizer. This is an opt-in
DINO adaptation, not a complete DINOv2 implementation.

Reuse `data/holdout-v1` exactly; do not regenerate the split when changing the
training seed. On Oscar, from the project directory:

```bash
# Review the five commands without submitting.
DRY_RUN=1 bash submit_dino_followups.sh --constraint=l40s
# Submit five jobs, seed 42.
bash submit_dino_followups.sh --constraint=l40s
```

For Hydra, use `--partition=gpus` and an available feature constraint:

```bash
bash submit_dino_followups.sh --partition=gpus --constraint='l40s|l40|rtx_a6000'
```

Repeat only promising settings with the second training seed, for example:

```bash
SEED=123 EXPERIMENT=global-crop bash submit_dino_followups.sh --constraint=l40s
```

The full experiment names above, plus `baseline`, are accepted by `EXPERIMENT`.
`all` is the default and deliberately omits another baseline training run.
`SPLIT_DIR` defaults to `data/holdout-v1`.

### Probe outputs and comparison rules

The jobs retain the existing official ridge implementation and support/query
protocol: SSL sees 6,001 training images, the ridge probe sees their 601 labelled
examples, and the monitoring query set has 149 labelled unseen images. Query
labels never fit the ridge weights, standardization, or regularization strength.
All SSL datasets expose only images to the training objective.

* `holdout/final_accuracy` is the primary comparison, at epoch 500.
* `holdout/final_macro_accuracy` and `holdout/domain/{galaxy,natural,satellite}_accuracy`
  diagnose which distinctions changed. Domain class groups are manually inferred
  from inspected examples, **not official domain metadata**. They are never used
  to alter SSL sampling, losses, or the official classifier.
* `holdout/best_accuracy` remains available for exploration; it is not the primary
  score after repeatedly checking the same small validation set.
* At the final epoch only, ten fixed draws of five support images per class fit
  separate ridge probes and classify the same held-out queries. These log
  `holdout/low_shot_5/accuracy_mean`, `accuracy_std`, and `macro_accuracy_mean`.
  The diagnostic sampling seed is **2026**, independent of training seed. The
  standard deviation measures support-draw variability, not a confidence interval.
* `probe-history.jsonl` saves monitoring reports. `probe-final.json` includes final
  per-class/domain results and low-support draw indices. `probe-final-embeddings.npz`
  caches support/query features when final low-support probing is enabled.
* `model.onnx` is the final teacher; `best.onnx` is the monitoring peak. Compare
  like with like and preserve file hashes when downloading/submitting.

One query image is 0.67 percentage points. Require replication across seeds and
inspect domain tradeoffs before promoting small gains. Low-support diagnostics
still use familiar classes; **they do not establish transfer to unseen classes**.
Full-data LOO, image-held-out accuracy, and online challenge scores are different
protocols and must not be ranked as if interchangeable.

Old baseline checkpoints need not be retrained just to obtain the new diagnostics.
For example, on Oscar evaluate the existing seed-42 holdout final model:

```bash
.venv/bin/python diagnose.py runs/dino-resnet-9113501/model.onnx \
  --data data/holdout-v1/holdout.parquet \
  --support-data data/holdout-v1/train.parquet \
  --low-shot 5 --repeats 10 --seed 2026 --threads 4 \
  --save-embeddings --out runs/diagnostics-holdout-9113501
```

The matching seed-123 baseline is job `9124171`. Replace both job IDs in that
command to produce its report. Run on an allocated CPU node; no GPU is required.
Standalone diagnostics without `--support-data` retain reference LOO. Optional
low-support diagnostics on a single file exclude sampled support rows from that
repeat's query set, but do not imply those images were excluded from SSL training.

After selecting a candidate, check split sensitivity with fresh baseline and
candidate training on a second fixed split, **not another sweep**:

```bash
.venv/bin/python prepare_holdout.py --seed 2027 --out data/holdout-confirm
SPLIT_DIR=data/holdout-confirm EXPERIMENT=baseline \
  bash submit_dino_followups.sh --constraint=l40s
SPLIT_DIR=data/holdout-confirm EXPERIMENT=global-crop \
  bash submit_dino_followups.sh --constraint=l40s
```

Replace `global-crop` with the selected winner. This is a split-sensitivity check,
not a wholly independent untouched test set: it is drawn from the same original
dataset and can overlap the first split. Keep its seed fixed and do not tune on
it repeatedly. Train the eventual chosen recipe on all 7,500 images before
submitting a full-data model. The holdout-trained model is a diagnostic candidate.

New CLI flags are `--global-crop-min`, `--blur-sigma-max`, `--koleo-weight`,
`--probe-low-shot`, `--probe-repeats`, and `--probe-seed`. Resuming restores saved
augmentation/KoLeo settings and rejects conflicting explicit training flags.
These experiment changes require fresh runs.
New checkpoints also save and restore local crop count. Legacy checkpoints lack
that field, so supply their original `--local-crops` explicitly when resuming.

This experiment uses stable-pretraining 0.1.8's DINO method, loss, EMA teacher
callback, and training loop. New runs default to a randomly initialized compact
ResNet-18 (`--backbone resnet18_small`). It uses two residual blocks per stage,
32/64/128/256 channels, a 3×3 stride-1 stem without max pooling, and global
average pooling for 256-dimensional features. Half-width stages keep the
FP32 encoder below the challenge's 20 MiB limit; this is not standard-width
ResNet-18. BatchNorm floating statistics follow the EMA teacher update, while
integer batch counters are copied from the student.

Use `--backbone resnet18_wide` for a wider ResNet-18 with stages
40/80/160/320 and 320-dimensional features. Its verified FP32 ONNX export is
16.66 MiB, below the 20 MiB limit. It keeps the same blocks, stem,
pooling, and DINO training method. This is a fresh architecture experiment,
so start from scratch rather than resuming a width-32 checkpoint. Checkpoints
record the backbone selection for later resume or export. The smaller ResNet
remains the default, including in `dino.sbatch`.

For the wider experiment on Oscar's L40S GPUs:

```bash
sbatch --constraint=l40s dino.sbatch \
  --backbone resnet18_wide --epochs 300 --batch-size 64 \
  --lr 0.001 --weight-decay 0.01 --local-crops 2 --seed 42 \
  --wandb-name resnet18-wide-lr0.001-wd0.01-crops2-seed42
```

### Deeper compact ResNet experiment

Use `--backbone resnet34_small` to test ResNet-34 depth: 3/4/6/3 residual
blocks, stages 28/56/112/224, and 224-dimensional features. It retains the same
small-image stem, global average pooling, DINO head, BatchNorm/EMA handling,
and export pipeline. The verified FP32 ONNX export is 15.56 MiB, below 20 MiB.
This compares deeper/narrower against ResNet-18-wide, rather than isolating
depth at identical width. All encoders are randomly initialized.

Keep training settings matched to the 500-epoch wide-ResNet baseline:

```bash
sbatch --partition=gpu --gres=gpu:1 --constraint=l40s dino.sbatch \
  --backbone resnet34_small --epochs 500 --batch-size 64 \
  --lr 0.001 --weight-decay 0.01 --local-crops 2 --seed 42 \
  --wandb-name resnet34-small-500ep-lr0.001-wd0.01-crops2-seed42
```

Start a fresh run; ResNet-18 checkpoints cannot resume into ResNet-34.
The existing ResNet-18 and ViT defaults/checkpoints remain supported. Append
`--epochs 1` for an initial cluster check. Measure epoch time before assuming
the batch script's 3-hour allocation covers 500 epochs of the deeper model.

The previous ViT remains available with `--backbone vit_tiny`: 192-dimensional
features, 3 attention heads, 10 blocks, and 8×8 patches.
It is a starting baseline, not a tuned model or a pretrained DINOv2 model.
External pretrained weights remain an unresolved course-rule question.

### Shorter run with the longer schedule

`--epochs` controls when training stops. Optional `--schedule-epochs` sets the
learning-rate warmup/cosine horizon and teacher EMA horizon independently. It
also determines teacher-temperature warmup length (still capped at 30 epochs,
so both 300 and 500 have the same 30-epoch temperature warmup). Without this
option, existing runs retain their original schedule behavior.

```bash
sbatch --partition=gpu --gres=gpu:1 dino.sbatch \
  --backbone resnet18_wide --epochs 300 --schedule-epochs 500 \
  --batch-size 64 --local-crops 2 --lr 0.001 --weight-decay 0.01 --seed 42 \
  --wandb-name wide-300ep-schedule500-crops2-seed42
```

With the same batches/settings, this uses the first 300 epochs of the 500-epoch
LR and EMA schedules. It does not compress 500 epochs of learning into 300 or
guarantee the same accuracy. GPU/kernel randomness can still affect results.
The schedule horizon must be at least the training duration. It is recorded
in checkpoints and W&B. Resume restores it; an explicit conflicting override
is rejected. To resume this run to 500 epochs, keep `--schedule-epochs 500`
and the same batch size/data.

### Optional learning-rate and teacher-momentum search

`--teacher-momentum` sets the initial EMA coefficient for both encoder and
projection head; it follows the existing cosine schedule towards 1. Its
default remains 0.996. Values must be strictly between 0 and 1, and checkpoints
preserve the setting. Resume restores it rather than resetting the teacher.

`submit_dino_grid.sh` queues nine independent 500-epoch ResNet-18-wide runs:
LR `{0.0007, 0.001, 0.0015}` crossed with initial teacher EMA
`{0.996, 0.999, 0.9995}`. All use WD 0.01, two local crops, batch 64, and seed
42. It includes a repeat of the existing baseline. Larger EMA means a slower
teacher update, not a guaranteed improvement. This explores the teacher
dynamics rather than spending another full grid on crop count.

```bash
# Optional: submits nine jobs, three hours each, without restricting the GPU model.
bash submit_dino_grid.sh
# Or request specific hardware:
bash submit_dino_grid.sh --constraint=l40s
# Hydra partition override:
bash submit_dino_grid.sh --partition=gpus --constraint=l40s
```

Run the 300/500 schedule comparison first if compute is limited. Search results
at 300 epochs need not rank configurations the same as at 500; the helper uses
500 to match the established accuracy baseline. Confirm any winner with seed
123, compare final exported ONNX scores, and retain existing strong models.

New runs use a normalized prototype layer (fixed unit weight norms), truncated
normal initialization for the projection head, and freeze the prototype layer
during the first epoch. Biases and normalization parameters are excluded from
AdamW weight decay. The stable-pretraining DINO method still handles view
matching, teacher sharpening/centering, and student gradients; its EMA callback
updates both the teacher encoder and head after optimizer steps. The loss center
is included in checkpoints. This remains a compact DINO adaptation: fewer
prototypes, smaller crops/head, epoch-based EMA scheduling, and constant weight
decay differ from the original ImageNet recipe.

When resuming an older checkpoint, its original head and freeze settings are
preserved, including the ViT architecture for checkpoints predating backbone
selection. Backbone changes require a fresh run in a new output directory.

The random seed is fixed (default 42). The trainer uses `deterministic="warn"`
to prefer deterministic operations where available without aborting on the
ViT positional-embedding resize, whose bicubic CUDA backward has no
deterministic implementation. Training is not guaranteed to be bit-for-bit
identical across runs or machines.

Training uses two 96×96 global crops and six 48×48 local crops, ImageNet
normalization, AdamW, learning-rate warmup and cosine decay. Labels are removed
before sampling training examples. The parquet file must already exist locally;
the training script does not download data or weights. CSV logs stay local;
Weights & Biases logging is available with `--wandb`.

Install the locked environment and check the full pipeline:

```bash
uv sync
uv run main.py --smoke-test --output runs/dino-smoke
uv run test_submission.py runs/dino-smoke/model.onnx
```

The smoke test trains on four images for two CPU steps. Its embeddings are only
for checking the pipeline; they are not a meaningful trained submission.

Start training (the default data path is `data/train.parquet`):

```bash
uv run main.py --backbone resnet18_small --epochs 300 --batch-size 64 \
  --lr 0.0005 --seed 42 --local-crops 2 --output runs/dino-resnet
```

Use `--accelerator gpu --precision 16-mixed --workers 4` on a CUDA machine.
The default uses one available device with full precision. Lower the batch size
if memory is limited. The initial learning rate is 0.0005; tune it alongside
batch size. Use a new output directory for each independent experiment. Both
encoders initially use the same AdamW recipe for comparison; the ResNet recipe
has not been tuned.

Use `--weight-decay VALUE` to tune AdamW regularization for a fresh run
(default `0.04`; biases and normalization parameters remain excluded).
Resuming a checkpoint restores its saved optimizer settings.

## Oscar batch jobs

Copy `dino.sbatch` to the project directory on Oscar. Submit from that directory
after installing the environment and authenticating W&B:

```bash
uv sync --locked --python 3.12
uv run wandb login
sbatch dino.sbatch
```

The script requests one GPU of any type on the `gpu` partition, along with one
node, one training task, and six CPU cores. Four cores are
budgeted for the four data-loader workers; the main process uses up to two
OpenMP/MKL threads. Workers are subprocesses of the single training task, not
additional Slurm tasks. This is a CPU budget, not per-worker core pinning.
`--ntasks-per-node=1` matches the trainer's `devices=1`.
Training uses GPU acceleration and mixed precision.

Follow progress with `tail -f dino-JOB_ID.out` and check the queue with
`squeue -u "$USER"`. Outputs go to `runs/dino-resnet-JOB_ID/`.
To change the epoch count or use offline W&B logging, append training options:

```bash
sbatch dino.sbatch --epochs 10 --wandb-mode offline
```

The batch script selects ResNet, 300 epochs, learning rate 0.0005, seed 42, and
two local crops. The default time limit is 3 hours, including
periodic accuracy probes. This is a conservative initial allocation, not a
measured runtime estimate; adjust after observing completed epoch times on your
assigned GPU. An epoch-end checkpoint is saved throughout training. To request
a different limit, for example, use `sbatch --time=06:00:00 dino.sbatch`.

For an Oscar L40S, use `sbatch --constraint=l40s dino.sbatch`. For Hydra's
L40S node, override the partition and node:

```bash
sbatch --partition=gpus --gres=gpu:1 --nodelist=gpu2501 dino.sbatch
```

Append `--epochs 1` for a one-epoch check, or `--backbone vit_tiny` to select
the previous encoder. Overrides follow the script filename.

### CUDA driver compatibility

The lockfile selects official CUDA 12.6 builds of PyTorch 2.14.1 and torchvision
0.29.1 on Linux x86-64. This avoids the default PyPI Linux CUDA 13 build, which
fails on Oscar nodes whose NVIDIA driver reports CUDA 12.9 support. macOS
continues to use PyPI builds. A CUDA toolkit module does not update the driver.

After copying both the updated `pyproject.toml` and `uv.lock` to Oscar, run:

```bash
uv sync --locked --python 3.12
```

Inside a GPU allocation, check the installed build and initialize CUDA before
starting training:

```bash
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); torch.cuda.init(); print(torch.cuda.get_device_name(0))'
```

The build should show `+cu126` and CUDA `12.6`. Actual GPU initialization must
be checked on Oscar; resolving the lockfile alone does not verify the node's
driver or GPU access.

## Weights & Biases

Authenticate once, then enable W&B logging:

```bash
uv run wandb login
uv run main.py --wandb --wandb-project dino-experiment --wandb-name resnet18-small-run1
```

Use `--wandb-entity YOUR_TEAM` to choose a team or account. W&B receives the
step/epoch training loss, learning rate, epoch/step counters, model
hyperparameters, batch size, crop settings, seed, and precision. Local CSV
logging continues alongside W&B. Checkpoints, image data, and the ONNX model
are not uploaded by this integration.

### Accuracy monitoring

The script logs `probe/loo_accuracy`, `probe/ridge_lambda`, labelled image/class
counts, and probe duration. Accuracy ranges from 0 to 1.
Monitoring runs after epoch 1, every 5 epochs,
and at the final epoch. Set `--probe-every 1` for every epoch or
`--probe-every 0` to disable it.

Each monitoring pass exports a separate copy of the EMA encoder through the
same ONNX path used for final submission, then calls `evaluate.py`'s `evaluate`
function on CPU. It uses the evaluator's exact input normalization, feature
standardization, regularization selection, and leave-one-out predictions on
all labelled training images. Only ONNX Runtime's CPU thread count is bounded
to the Slurm allocation (two threads outside Slurm). The temporary model is
deleted after scoring. Monitoring preserves the training RNG and live encoder.

The previous 80/20 metrics have been replaced, not renamed: this is a different
calculation. This now matches the published local evaluation protocol; the
organizers' hidden evaluation still uses different images/classes. Labels only
enter the separate CPU ridge probe, never DINO's training objective. At least
two labelled images are required. No classifier is exported in the submission.

Check W&B logging without logging in or contacting W&B:

```bash
uv run main.py --smoke-test --wandb --wandb-mode offline --output runs/dino-wandb-smoke
```

Offline run files are saved under the output directory's `wandb/` folder.
Training without `--wandb` uses only local CSV logging, which is suitable for
offline reproduction. Export-only commands do not create W&B runs.

Resume with the same training settings and output directory:

```bash
uv run main.py --epochs 100 --batch-size 64 --output runs/dino \
  --resume runs/dino/checkpoints/last.ckpt
```

To continue the same online W&B run, also pass `--wandb --wandb-id RUN_ID`
and the same W&B project/entity. The run ID appears in W&B's run URL; the
display name is not the run ID. Without an ID, W&B starts a new run.

Export again without training:

```bash
uv run main.py --export-checkpoint runs/dino/checkpoints/last.ckpt --output runs/dino
```

Checkpoints include teacher, student, optimizer, scheduler, and loss-center
state. The exported `model.onnx` contains only the EMA teacher encoder, with
256 small-ResNet-18 features, 320 wide-ResNet-18 features,
224 compact-ResNet-34 features, or 192 ViT features,
zero-padded to 1024. Export verifies agreement with PyTorch and
checks the 20 MiB limit. Run the challenge validator before any upload:

```bash
uv run test_submission.py runs/dino/model.onnx
uv run evaluate.py runs/dino/model.onnx --data data/train.parquet
```

The second command reports ridge-probe leave-one-out accuracy on the labelled
training examples; it is a monitoring score, not the hidden evaluation score.
No command here pushes code or uploads a submission.

## Generalization and augmentation experiments

The official `evaluate.py` remains unchanged. Local training-image LOO, a
diagnostic image holdout, and the hidden challenge score answer different
questions; none guarantees a particular value for the others.

### Inspect an exported model

Run this in a CPU allocation. It uses the official embedding normalization and
ridge implementation, reporting identical overall LOO accuracy plus per-class
accuracy, macro accuracy, a confusion matrix, and original row IDs of errors:

```bash
.venv/bin/python diagnose.py runs/dino-resnet-9061570/model.onnx \
  --data data/train.parquet --out runs/diagnostics-9061570 --threads 4
```

`diagnostics.json` includes model/data SHA-256 hashes to identify exactly which
artifact was evaluated. `mistakes.png` shows up to 24 mistakes with true and
predicted class IDs. Macro accuracy weights classes equally; the official score
continues to weight images equally. These labels are used only in diagnostics.

### Prepare a fixed unseen-image diagnostic

Create the split once and reuse it for every candidate:

```bash
.venv/bin/python prepare_holdout.py --out data/holdout-v1 --seed 2026 --fraction 0.2
```

This writes `train.parquet`, `holdout.parquet`, and `split.json`. The manifest
contains original row indices and SHA-256 hashes. The two image partitions
are disjoint. Labels stratify the diagnostic partition, not the DINO objective;
training batches still contain only augmented images. Existing split directories
are not overwritten. Copy the same files/manifest across clusters for comparisons.

Fresh diagnostic encoders train on the approximately 80% training partition only.
A separate CPU ridge probe fits labelled training-partition embeddings (including
its usual LOO selection of lambda), and predicts the held-out labelled images.
It never uses query labels to fit or choose lambda. The logs use
`holdout/accuracy` and `holdout/macro_accuracy`, rather than `probe/loo_accuracy`.
This tests unseen images from supported classes; it does not simulate unseen
classes or domains in the private benchmark. Splitting data after an existing
encoder saw every image does not make its score an unseen-image result.

### Queue four controlled ablations

```bash
# Oscar: four new jobs, each requesting one GPU and three hours.
bash submit_dino_augmentations.sh --holdout --constraint=l40s
# Hydra alternative:
bash submit_dino_augmentations.sh --holdout --partition=gpus \
  --constraint="l40s|l40|rtx_a6000"
```

All four use wide ResNet-18, 500 epochs with a 500-epoch schedule, LR 0.0015,
weight decay 0.01, teacher momentum 0.996, batch 64, two local crops, seed 42.
The jobs are a matched baseline and one change each:

| Run | Change |
| --- | --- |
| baseline | Original augmentations |
| larger-local | Local crop area 0.15–0.4 instead of 0.05–0.4 |
| gentler-color | Half the original color-jitter strength |
| no-solarize | Solarization probability zero |

Global crop scale stays 0.4–1.0; grayscale, blur, and flips stay unchanged.
Individual CLI flags are `--local-crop-min`, `--local-crop-max`,
`--color-jitter-strength`, and `--solarize-probability`. Defaults exactly retain
the previous policy. Values are logged to W&B and stored in checkpoints. Resume
restores them and rejects explicit conflicting augmentation overrides. Continue
to supply the same training/probe files, crop count, and batch size when resuming.

Compare the four `holdout/accuracy` curves against their new matched baseline,
not against the full-data 74.4% LOO. Test a promising policy with seed 123.
For a final submission candidate, train a fresh encoder on all `data/train.parquet`
with the selected augmentation flags and the original monitoring protocol.
Omitting `--holdout` from the helper instead queues all four on the full dataset.

### Keep peak and final models

When probes are enabled, new runs retain `best.onnx` and `best.json` as well as
the final `model.onnx` and `checkpoints/last.ckpt`. Best refers to the configured
local monitoring metric, not the hidden benchmark. `best.json` records the
epoch, score, and protocol; the best ONNX is kept while the temporary probe
export is available. Best exports are encoders, not resumable training checkpoints.
A continuation in a new output directory tracks its own best in that segment;
the original run's best stays in its original directory. Check both candidates
with `test_submission.py` and `evaluate.py` before deciding what to submit.

## CPU-only intermediate-feature comparison

`compare_feature_exports.py` loads one existing compact-ResNet checkpoint and
exports both the usual final-stage embedding and a multiscale embedding from
the exact same teacher weights. Multiscale concatenates global-average-pooled
outputs of stages 2, 3, and 4. ResNet-18-wide produces 80 + 160 + 320 = 560
features, zero-padded to 1024. It has no extra trained parameters and requires
no retraining. Its value is an experimental hypothesis, not a promised gain.
The default training/monitoring export remains the original final-stage version.

Inside an Oscar CPU allocation:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

# Compare the full-data model against its own ordinary export.
.venv/bin/python compare_feature_exports.py \
  runs/dino-resnet-9061570/checkpoints/last.ckpt \
  --data data/train.parquet --out runs/features-9061570 --threads 4

# Independently compare the matched holdout-baseline encoder.
.venv/bin/python compare_feature_exports.py \
  runs/dino-resnet-9113501/checkpoints/last.ckpt \
  --data data/holdout-v1/holdout.parquet \
  --support-data data/holdout-v1/train.parquet \
  --out runs/features-9113501 --threads 4
```

Each comparison saves `final/model.onnx`, `multiscale/model.onnx`, per-class
`diagnostics.json` reports, mistake sheets, and `comparison.json`. It prints
changes in image-weighted and macro accuracy. The reports contain file hashes.
Existing output directories are rejected to protect earlier artifacts. These
comparisons use final training checkpoints, not the independently saved peak
`best.onnx`, which is not a resumable PyTorch checkpoint.

Look for gains on both protocols, including the previously weak galaxy classes,
before selecting an export. Do not compare the holdout percentage directly with
full-data LOO. Both final and multiscale ONNX files must still fit 20 MiB; export
checks PyTorch/ONNX agreement and size. Run `test_submission.py` on any candidate
before uploading it. The script neither logs a W&B run nor submits anything.

For exporting a single view without evaluation:

```bash
.venv/bin/python main.py \
  --export-checkpoint runs/dino-resnet-9061570/checkpoints/last.ckpt \
  --export-features multiscale --output runs/multiscale-9061570
```

`--export-features multiscale` is export-only and rejects ViT checkpoints.

For the matched seed-123 baseline vs gentler-color repeat on the existing split:

```bash
bash submit_color_repeat.sh --constraint=l40s
```

This submits two new 500-epoch jobs, three hours each, differing only in color
jitter strength (1.0 vs 0.5). The training seed changes to 123, but the data split
stays `data/holdout-v1`; do not make a new partition for the new seed. Compare
both at completed epochs 370 and 500 as well as their separately recorded peaks.

## Optional right-angle rotation ablation

`--rotate90-probability` defaults to 0. When disabled, no rotation transform is
inserted and no extra random numbers are consumed, preserving the earlier
augmentation pipeline. Older checkpoints without the field restore probability
0. Resume restores the saved probability and rejects explicit conflicting values.
Existing submission helpers continue to use rotation disabled.

```bash
# Oscar: one additional job; original colors/crops with rotation enabled.
bash submit_rotation.sh --constraint=l40s
# Hydra alternative:
bash submit_rotation.sh --partition=gpus --constraint="l40s|l40|rtx_a6000"
```

This creates a fresh seed-42, 500-epoch holdout experiment matched to the existing
seed-42 baseline. Each global and local crop independently has a 50% chance of
being rotated; a selected crop gets 90, 180, or 270 degrees with equal probability.
PIL transpose performs exact right-angle rotations without interpolation or
filled corners. The change applies to training images only; evaluator inputs
and submission preprocessing are unchanged. The value is logged to W&B and saved
in checkpoint augmentation settings. The helper requests one GPU for three hours,
uses the existing data/holdout-v1 partition, and retains original color strength
and solarization. Compare its holdout and macro accuracy with the baseline before
combining rotation with another augmentation or retraining on the full dataset.

## SGD optimization-recipe experiment

`--optimizer sgd` opts into PyTorch SGD with momentum. `--sgd-momentum` defaults
to 0.9 for SGD. Existing commands and checkpoints retain AdamW by default.
Optimizer type and momentum are recorded in checkpoints and W&B. Resume restores
the saved optimizer and rejects a conflicting optimizer or momentum request;
changing optimizers requires a fresh run. Bias/norm parameters remain excluded
from decay and frozen teacher parameters remain outside the optimizer.

The search compares three actual peak learning rates: 0.005, 0.01, and 0.02,
all with SGD momentum 0.9 and weight decay 0.0001. This project does not scale
these LRs by batch size. They are an exploratory range around the original DINO
CNN recipe's scale, not an exact reproduction of its ImageNet training. This
is a recipe comparison (optimizer plus appropriate LR/decay), not an isolated
optimizer-only ablation. The original recipe is documented at
https://github.com/facebookresearch/dino#training.

```bash
# Oscar: three new jobs, one GPU and three hours each.
bash submit_sgd_search.sh --constraint=l40s
# Hydra alternative, using the same fixed split:
bash submit_sgd_search.sh --partition=gpus --constraint="l40s|l40|rtx_a6000"
```

All use ResNet-18-wide, 500 training/schedule epochs, seed 42, batch 64, two
local crops, teacher momentum 0.996, and original augmentations with rotation
disabled. The training/probe files are the existing data/holdout-v1 partitions.
No changes are made to existing batch scripts or earlier runs. Compare the
500-epoch holdout and macro accuracies against the matched AdamW seed-42
baseline (69.13% and 66.27%); keep peak comparisons separate. Repeat a promising
SGD recipe with seed 123 and compare against that AdamW baseline (72.48% holdout,
68.94% macro). The validated full-data 74.4% LOO has a different protocol and
training-set size. Train on the full dataset only after choosing a supported
recipe; keep the established AdamW submission candidate.

For a one-epoch cluster check before queuing the full search:

```bash
sbatch --constraint=l40s --time=00:20:00 dino.sbatch \
  --backbone resnet18_wide --optimizer sgd --sgd-momentum 0.9 \
  --lr 0.01 --weight-decay 0.0001 --epochs 1 --schedule-epochs 500 \
  --batch-size 64 --local-crops 2 --seed 42 \
  --data data/holdout-v1/train.parquet \
  --probe-data data/holdout-v1/holdout.parquet \
  --probe-support-data data/holdout-v1/train.parquet \
  --wandb-name sgd-holdout-one-epoch-check
```
