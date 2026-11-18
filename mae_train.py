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

MAE + SIGReg hybrid (--sigreg-weight > 0): each image gets --views augmented views. Besides the MAE loss
on every (masked) view, the UNMASKED encoder embeds every view (the same embedding we submit), a small
projector maps it to --proj-dim, and LeJEPA's loss is added:

    loss = recon_weight * MAE  +  sigreg_weight * (invariance + sigreg_lambda * SIGReg)

  invariance  views of the same image should land on the same point (mean squared distance to their centre)
  SIGReg      all embeddings together should look like an isotropic Gaussian (sliced Epps-Pulley test),
              which stops the invariance term from collapsing everything to one point

--recon-weight 0 gives plain LeJEPA on the same encoder, for comparison.
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
from stable_pretraining.methods.lejepa import SlicedEppsPulley
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
    p.add_argument("--jitter", type=float, default=0.0, help="probability of colour jitter (0.4, 0.4, 0.2, 0.1)")
    p.add_argument("--gray", type=float, default=0.0, help="probability of converting a view to grayscale")
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
    p.add_argument("--feature-layers", default="",
                   help="comma-separated block numbers to pool and concatenate, e.g. 7,8,9 (default: final output)")
    # MAE + SIGReg hybrid (LeJEPA's loss on top of MAE). --sigreg-weight 0 (default) = pure MAE.
    p.add_argument("--sigreg-weight", type=float, default=0.0,
                   help="weight of the LeJEPA term (invariance + lambda * SIGReg); 0 disables it")
    p.add_argument("--recon-weight", type=float, default=1.0, help="weight of the MAE loss (0 = LeJEPA only)")
    p.add_argument("--sigreg-lambda", type=float, default=0.02, help="SIGReg weight inside the LeJEPA term")
    p.add_argument("--views", type=int, default=2, help="augmented views per image (hybrid only)")
    p.add_argument("--proj-hidden", type=int, default=1024)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--slices", type=int, default=1024, help="random directions for the SIGReg test")
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
    """Training view(s) of one image: random crop (resized back to 96) + horizontal flip, optional colour
    jitter / grayscale, then normalise. With views > 1 it returns a tuple of independently augmented views."""

    def __init__(self, images, crop_min, views=1, jitter=0.0, gray=0.0):
        self.images, self.views = images, views
        tfs = [
            v2.RandomResizedCrop(96, scale=(crop_min, 1.0), interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(),
        ]
        if jitter > 0:
            tfs.append(v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=jitter))
        if gray > 0:
            tfs.append(v2.RandomGrayscale(p=gray))
        tfs += [
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float [0,1]
            v2.Normalize(MEAN, STD),                 # same ImageNet normalisation the evaluator applies
        ]
        self.tf = v2.Compose(tfs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.views == 1:
            return self.tf(self.images[i])
        return tuple(self.tf(self.images[i]) for _ in range(self.views))


def normalise(images_uint8):
    """Exactly the evaluator's preprocessing: [0,1] then ImageNet mean/std. No augmentation."""
    x = images_uint8.float() / 255.0
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

def pool_tokens(tokens, pool):
    """[B, 1 + num_patches, D] tokens -> one vector per image."""
    cls, patches = tokens[:, 0], tokens[:, 1:]
    if pool == "cls":
        return cls
    if pool == "mean":
        return patches.mean(1)
    return torch.cat([cls, patches.mean(1)], dim=1)


def block_outputs(vit, x):
    """Tokens after every transformer block (before the final LayerNorm), as a list of length depth."""
    x = vit.patch_embed(x)
    x = vit._pos_embed(x)  # prepend CLS token, add positional embedding
    x = vit.norm_pre(vit.patch_drop(x))
    outs = []
    for blk in vit.blocks:
        x = blk(x)
        outs.append(x)
    return outs


class Encoder(nn.Module):
    """What we submit: the ViT (no masking at test time) followed by pooling into one vector.

    layers=() uses the final output (after the last block and LayerNorm). layers=(7, 8, 9) instead pools
    the outputs of blocks 7, 8 and 9 (1-indexed) separately and concatenates them.
    """

    def __init__(self, vit, pool="mean", layers=()):
        super().__init__()
        self.vit, self.pool, self.layers = vit, pool, tuple(layers)

    def forward(self, x):
        if not self.layers:
            return pool_tokens(self.vit.forward_features(x), self.pool)  # [B, 1 + 144, D]
        outs = block_outputs(self.vit, x)
        return torch.cat([pool_tokens(outs[i - 1], self.pool) for i in self.layers], dim=1)


def parse_layers(text):
    return tuple(int(i) for i in text.split(",") if i.strip())


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


def build_projector(in_dim, hidden, out_dim):
    """Small MLP head used only by the LeJEPA term during training (not exported)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, out_dim),
    )


def lejepa_loss(z, sigreg, lamb):
    """LeJEPA loss on projections z [views, batch, dim]: invariance + lamb * SIGReg (as in stable-pretraining)."""
    center = z.mean(0)
    inv = (z - center).square().mean()
    sig = sigreg(z.reshape(-1, z.size(-1)))
    return inv + lamb * sig, inv, sig


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

    hybrid = a.sigreg_weight > 0
    loader = DataLoader(
        Augmented(images, a.crop_min, a.views if hybrid else 1, a.jitter, a.gray), batch_size=a.batch_size, shuffle=True, drop_last=True,
        num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(a.seed),
    )

    mae, vit = build_mae(a)
    mae.to(device)
    encoder = Encoder(vit, a.pool, parse_layers(a.feature_layers))
    n_enc = sum(p.numel() for p in vit.parameters())
    n_dec = sum(p.numel() for p in mae.parameters()) - n_enc
    print(f"encoder {n_enc / 1e6:.2f}M params (~{n_enc * 4 / 2**20:.1f} MiB in ONNX), decoder {n_dec / 1e6:.2f}M")

    heads = nn.ModuleDict()  # training-only parts of the LeJEPA term: projector + SIGReg test
    if hybrid:
        with torch.no_grad():
            feat_dim = encoder(torch.zeros(2, 3, 96, 96, device=device)).shape[1]
        heads["projector"] = build_projector(feat_dim, a.proj_hidden, a.proj_dim)
        heads["sigreg"] = SlicedEppsPulley(num_slices=a.slices)
        heads.to(device)
        print(f"MAE + SIGReg hybrid: {a.views} views, loss = {a.recon_weight} * MAE + {a.sigreg_weight} * "
              f"(invariance + {a.sigreg_lambda} * SIGReg), projector {feat_dim}-{a.proj_hidden}-{a.proj_dim}")

    opt = torch.optim.AdamW(param_groups(mae, a.weight_decay) + param_groups(heads, a.weight_decay),
                            lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = a.epochs * steps_per_epoch, a.warmup_epochs * steps_per_epoch

    # Resume if a checkpoint exists, otherwise record this run's untrained floor
    ckpt_path, start_epoch = os.path.join(a.out, "last.pt"), 0
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        mae.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        if "heads" in ck:
            heads.load_state_dict(ck["heads"])
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
        mae.train(); heads.train()
        t0, sums = time.time(), {"loss": 0.0, "recon": 0.0, "inv": 0.0, "sigreg": 0.0}
        for i, x in enumerate(loader):
            step = epoch * steps_per_epoch + i
            lr = lr_at(step, total_steps, warmup_steps, a.lr, a.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            if not hybrid:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = mae(x).loss
            else:
                xs = [v.to(device, non_blocking=True) for v in x]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    recon = (torch.stack([mae(v).loss for v in xs]).mean() if a.recon_weight > 0
                             else torch.zeros((), device=device))
                    z = heads["projector"](encoder(torch.cat(xs)))  # unmasked: the embedding we submit
                z = z.float().view(len(xs), -1, z.size(-1))  # the LeJEPA term in fp32 for stability
                jepa, inv, sig = lejepa_loss(z, heads["sigreg"], a.sigreg_lambda)
                loss = a.recon_weight * recon.float() + a.sigreg_weight * jepa
                sums["recon"] += recon.item(); sums["inv"] += inv.item(); sums["sigreg"] += sig.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sums["loss"] += loss.item()

        rec = {"epoch": epoch + 1, "loss": sums["loss"] / steps_per_epoch}
        if hybrid:
            rec.update({k: sums[k] / steps_per_epoch for k in ("recon", "inv", "sigreg")})
        rec.update({"lr": lr, "sec": round(time.time() - t0, 1)})
        if (epoch + 1) % a.probe_every == 0 or epoch + 1 == a.epochs:
            rec["probe"] = probe_accuracy(encoder, probe_images, probe_labels, device)
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

        tmp = ckpt_path + ".tmp"  # write then rename, so a job killed mid-save never corrupts the checkpoint
        torch.save({"model": mae.state_dict(), "heads": heads.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "args": vars(a)}, tmp)
        os.replace(tmp, ckpt_path)

    export_onnx(encoder, os.path.join(a.out, "model.onnx"))
    print(f"done: {a.out}/model.onnx")


if __name__ == "__main__":
    main()
