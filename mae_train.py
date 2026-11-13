"""MAE (masked autoencoder) pretraining for the image challenge: baseline script.

    uv run mae_train.py --out runs/mae-baseline                  # full run with default settings
    uv run mae_train.py --smoke --out runs/mae-smoke             # 2 tiny epochs, checks the whole pipeline

What it does, in order:
  1. Loads all 7,500 images from train.parquet into memory. Labels are kept aside and used ONLY for
     monitoring (the ridge probe below), never in the loss.
  2. Builds a small ViT encoder (ViT-Tiny width, patch 8 -> 12x12 = 144 patches per 96px image) and
     wraps it in stable-pretraining's MAE: random 75% of patches are hidden, the encoder sees the
     rest, a light decoder reconstructs the hidden patches' (per-patch normalised) pixels.
  3. Trains with AdamW, linear warmup then cosine decay, mixed precision on GPU.
  4. Every --probe-every epochs, embeds the 750 labelled images (no masking, no augmentation) and
     reports the same leave-one-out ridge probe accuracy as evaluate.py.
  5. Saves a checkpoint every epoch (resumes automatically from it), and at the end exports only the
     encoder (decoder thrown away) to <out>/model.onnx in the submission format.

It also exports the untrained encoder to <out>/model_init.onnx, so every run has its own floor.
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from evaluate import UNLABELLED, Ridge
from export_onnx import MEAN, STD, export_onnx
from stable_pretraining.methods.mae import MAE
from training_data import load_training_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/train.parquet")
    p.add_argument("--out", default="runs/mae")
    p.add_argument("--seed", type=int, default=0)
    # Training schedule
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (after warmup)")
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.05)
    # Augmentation (MAE needs very little: crop + flip)
    p.add_argument("--crop-min", type=float, default=0.2, help="smallest RandomResizedCrop area fraction")
    # MAE
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--no-norm-pix", action="store_true", help="reconstruct raw pixels instead of per-patch normalised ones")
    # Encoder (this is what gets submitted: must stay under 20 MiB)
    p.add_argument("--patch-size", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--heads", type=int, default=3)
    # Decoder (thrown away after training, so its size is free)
    p.add_argument("--dec-dim", type=int, default=256)
    p.add_argument("--dec-depth", type=int, default=4)
    p.add_argument("--dec-heads", type=int, default=8)
    # How an image's tokens become one embedding vector
    p.add_argument("--pool", choices=["mean", "cls", "cls+mean"], default="mean")
    # Plumbing
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on GPU")
    p.add_argument("--smoke", action="store_true", help="2 epochs on 512 images, to test the pipeline")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

def load_images(path):
    """All images as one uint8 tensor [N, 3, 96, 96], plus the labels (-1 = unlabelled)."""
    ds = load_training_data(path)
    x = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in ds["image"]])
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous(), np.asarray(ds["label"])


class Augmented(Dataset):
    """Training view of one image: random crop (resized back to 96) + horizontal flip, then normalise."""

    def __init__(self, images, crop_min):
        self.images = images
        self.tf = v2.Compose([
            v2.RandomResizedCrop(96, scale=(crop_min, 1.0), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
            v2.Normalize(MEAN, STD),                 # same ImageNet normalisation the evaluator applies
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i])


def normalise(images_uint8):
    """Exactly the evaluator's preprocessing: [0,1] then ImageNet mean/std. No augmentation."""
    x = images_uint8.float() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    """What we submit: the ViT (no masking at test time) followed by pooling into one vector."""

    def __init__(self, vit, pool="mean"):
        super().__init__()
        self.vit, self.pool = vit, pool

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + 144, D]: CLS token, then one token per patch
        cls, patches = tokens[:, 0], tokens[:, 1:]
        if self.pool == "cls":
            return cls
        if self.pool == "mean":
            return patches.mean(1)
        return torch.cat([cls, patches.mean(1)], dim=1)


def build_mae(a):
    vit = VisionTransformer(
        img_size=96, patch_size=a.patch_size, embed_dim=a.embed_dim, depth=a.depth,
        num_heads=a.heads, num_classes=0,
    )
    mae = MAE(
        vit,
        decoder_embed_dim=a.dec_dim, decoder_depth=a.dec_depth, decoder_num_heads=a.dec_heads,
        mask_ratio=a.mask_ratio, norm_pix_loss=not a.no_norm_pix,
    )
    return mae, vit


def param_groups(model, weight_decay):
    """Weight decay on weight matrices only; not on biases, norms, positional embeddings or tokens."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in name for k in ("pos_embed", "cls_token", "mask_token")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, total, warmup, peak, floor):
    """Linear warmup to `peak`, then cosine decay down to `floor`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------------------------------
# Monitoring: same ridge probe as evaluate.py, labels used here only
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def probe_accuracy(encoder, images, labels, device, batch=256):
    was_training = encoder.training
    encoder.eval()
    z = torch.cat([encoder(normalise(images[i:i + batch]).to(device)).float().cpu()
                   for i in range(0, len(images), batch)]).numpy().astype(np.float64)
    encoder.train(was_training)
    return float((Ridge().fit(z, labels).loo_pred == labels).mean())


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------

def main():
    a = parse_args()
    if a.smoke:
        a.epochs, a.warmup_epochs, a.probe_every, a.batch_size = 2, 1, 1, 64
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not a.no_amp

    images, labels = load_images(a.data)
    if a.smoke:
        keep = np.r_[np.flatnonzero(labels != UNLABELLED)[:200], np.flatnonzero(labels == UNLABELLED)[:312]]
        images, labels = images[keep], labels[keep]
    lab = labels != UNLABELLED
    probe_images, probe_labels = images[lab], labels[lab]
    print(f"{len(images)} training images ({lab.sum()} labelled, used only for the monitoring probe)")

    loader = DataLoader(
        Augmented(images, a.crop_min), batch_size=a.batch_size, shuffle=True, drop_last=True,
        num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(a.seed),
    )

    mae, vit = build_mae(a)
    mae.to(device)
    encoder = Encoder(vit, a.pool)
    n_enc = sum(p.numel() for p in vit.parameters())
    n_dec = sum(p.numel() for p in mae.parameters()) - n_enc
    print(f"encoder {n_enc / 1e6:.2f}M params (~{n_enc * 4 / 2**20:.1f} MiB in ONNX), decoder {n_dec / 1e6:.2f}M")

    opt = torch.optim.AdamW(param_groups(mae, a.weight_decay), lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = a.epochs * steps_per_epoch, a.warmup_epochs * steps_per_epoch

    # Resume if a checkpoint exists, otherwise record this run's untrained floor
    ckpt_path, start_epoch = os.path.join(a.out, "last.pt"), 0
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        mae.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {ckpt_path} at epoch {start_epoch}")
    else:
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(vars(a), f, indent=2)
        acc0 = probe_accuracy(encoder, probe_images, probe_labels, device)
        print(f"epoch 0 (untrained) probe {100 * acc0:.1f}")
        export_onnx(encoder, os.path.join(a.out, "model_init.onnx"))
        mae.to(device)  # export_onnx moves the encoder to CPU

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    for epoch in range(start_epoch, a.epochs):
        mae.train()
        t0, loss_sum = time.time(), 0.0
        for i, x in enumerate(loader):
            step = epoch * steps_per_epoch + i
            lr = lr_at(step, total_steps, warmup_steps, a.lr, a.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = mae(x).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()

        rec = {"epoch": epoch + 1, "loss": loss_sum / steps_per_epoch, "lr": lr, "sec": round(time.time() - t0, 1)}
        if (epoch + 1) % a.probe_every == 0 or epoch + 1 == a.epochs:
            rec["probe"] = probe_accuracy(encoder, probe_images, probe_labels, device)
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

        tmp = ckpt_path + ".tmp"  # write then rename, so a job killed mid-save never corrupts the checkpoint
        torch.save({"model": mae.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "args": vars(a)}, tmp)
        os.replace(tmp, ckpt_path)

    export_onnx(encoder, os.path.join(a.out, "model.onnx"))
    print(f"done: {a.out}/model.onnx")


if __name__ == "__main__":
    main()
