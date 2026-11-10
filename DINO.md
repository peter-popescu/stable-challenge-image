# DINO experiment

This experiment uses stable-pretraining 0.1.8's DINO method, loss, EMA teacher
callback, and training loop. The encoder is a randomly initialized ViT with
192-dimensional features, 3 attention heads, 10 blocks, and 8×8 patches.
It is a starting baseline, not a tuned model or a pretrained DINOv2 model.
External pretrained weights remain an unresolved course-rule question.

## MobileNet backbone comparison

Select `--backbone mobilenet_v3_large` to train a randomly initialized
torchvision MobileNetV3-Large with its classifier removed. Global average
pooling produces 960-dimensional features, zero-padded to 1024 in ONNX. The
ViT remains the default (`--backbone vit_tiny`). Both encoders use the same
stable-pretraining DINO objective, head, global/local crops, and CPU ONNX
leave-one-out probe. MobileNet BatchNorm running statistics follow the library's
EMA update; integer batch counters are copied from the student without averaging.

For a full one-epoch GPU check on Oscar:

```bash
sbatch --constraint=l40s dino-mobilenet.sbatch --epochs 1
```

For the 300-epoch comparison, using batch size 64, learning rate 0.0005, seed 42,
and four loader workers:

```bash
sbatch --constraint=l40s dino-mobilenet.sbatch
```

Omit `--constraint=l40s` to accept any GPU type. The script requests one GPU,
one task, six CPU cores, 32 GB RAM, and 12 hours. Logs are written to
`dino-mobilenet-JOB_ID.out` and outputs to `runs/dino-mobilenet-JOB_ID/`.
Run `uv sync --locked --python 3.12` before submitting after pulling updates.
To run directly inside an allocated GPU session:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python main.py --backbone mobilenet_v3_large --epochs 1 \
  --accelerator gpu --precision 16-mixed --batch-size 64 --workers 4 \
  --wandb --wandb-mode offline --output runs/mobilenet-check
```

New checkpoints record the architecture and export it automatically. Resume
restores that architecture; old checkpoints without a backbone field remain
ViTs. Start a fresh run when switching architectures. This comparison is not
a claim that the same optimization settings are ideal for both encoders.

The local full-epoch check completed 117 batches, saved/restored a checkpoint,
logged the CPU LOO probe offline, and exported an 11.33 MiB ONNX model. It was
tested on the Mac GPU in full precision, not Oscar CUDA/mixed precision.
The one-epoch model passed structural submission checks but triggered the
validator's low-embedding-variation warning. BatchNorm counters/statistics were
verified to update; the stock MobileNet BatchNorm momentum is 0.01, so early
evaluation statistics adapt slowly. Warmup is a possible contributor, not a
guarantee that the warning will resolve. Inspect accuracy and feature variation
after several epochs before treating the CNN as a successful quality baseline.

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
preserved. Start a fresh run in a new output directory to use the revised head.

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
uv run main.py --epochs 100 --batch-size 64 --output runs/dino
```

Use `--accelerator gpu --precision 16-mixed --workers 4` on a CUDA machine.
The default uses one available device with full precision. Lower the batch size
if memory is limited. The initial learning rate is 0.0005; tune it alongside
batch size. Use a new output directory for each independent experiment.

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
`squeue -u "$USER"`. Outputs go to `runs/dino-JOB_ID/`.
To change the epoch count or use offline W&B logging, append training options:

```bash
sbatch dino.sbatch --epochs 10 --wandb-mode offline
```

The default time limit is 12 hours for the 100-epoch training run, including
periodic accuracy probes. This is a conservative initial allocation, not a
measured runtime estimate; adjust after observing completed epoch times on your
assigned GPU. An epoch-end checkpoint is saved throughout training. To request
a different limit, for example, use `sbatch --time=06:00:00 dino.sbatch`.

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
uv run main.py --wandb --wandb-project dino-experiment --wandb-name vit-tiny8-run1
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
192 features zero-padded to 1024. Export verifies agreement with PyTorch and
checks the 20 MiB limit. Run the challenge validator before any upload:

```bash
uv run test_submission.py runs/dino/model.onnx
uv run evaluate.py runs/dino/model.onnx --data data/train.parquet
```

The second command reports ridge-probe leave-one-out accuracy on the labelled
training examples; it is a monitoring score, not the hidden evaluation score.
No command here pushes code or uploads a submission.
