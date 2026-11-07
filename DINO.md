# DINO experiment

This experiment uses stable-pretraining 0.1.8's DINO method, loss, EMA teacher
callback, and training loop. The encoder is a randomly initialized ViT with
192-dimensional features, 3 attention heads, 10 blocks, and 8×8 patches.
It is a starting baseline, not a tuned model or a pretrained DINOv2 model.
External pretrained weights remain an unresolved course-rule question.

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

The default time limit is one hour, matching the tested interactive allocation.
For a longer run, request more time, for example `sbatch --time=12:00:00 dino.sbatch`.

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
