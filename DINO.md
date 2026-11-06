# DINO experiment

This experiment uses stable-pretraining 0.1.8's DINO method, loss, EMA teacher
callback, and training loop. The encoder is a randomly initialized ViT with
192-dimensional features, 3 attention heads, 10 blocks, and 8×8 patches.
It is a starting baseline, not a tuned model or a pretrained DINOv2 model.
External pretrained weights remain an unresolved course-rule question.

Training uses two 96×96 global crops and six 48×48 local crops, ImageNet
normalization, AdamW, learning-rate warmup and cosine decay. Labels are removed
before sampling training examples. The parquet file must already exist locally;
the training script does not download data or weights. Logs stay local.

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

Resume with the same training settings and output directory:

```bash
uv run main.py --epochs 100 --batch-size 64 --output runs/dino \
  --resume runs/dino/checkpoints/last.ckpt
```

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
