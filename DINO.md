# DINO experiment

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

The previous ViT remains available with `--backbone vit_tiny`: 192-dimensional
features, 3 attention heads, 10 blocks, and 8×8 patches.
It is a starting baseline, not a tuned model or a pretrained DINOv2 model.
External pretrained weights remain an unresolved course-rule question.

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
two local crops. The default time limit is 12 hours, including
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
256 small-ResNet features, 320 wide-ResNet features, or 192 ViT features,
zero-padded to 1024. Export verifies agreement with PyTorch and
checks the 20 MiB limit. Run the challenge validator before any upload:

```bash
uv run test_submission.py runs/dino/model.onnx
uv run evaluate.py runs/dino/model.onnx --data data/train.parquet
```

The second command reports ridge-probe leave-one-out accuracy on the labelled
training examples; it is a monitoring score, not the hidden evaluation score.
No command here pushes code or uploads a submission.
