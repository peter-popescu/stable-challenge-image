"""Masked image modelling (SimMIM-style MAE objective) with the team's ResNet9 backbone.

    uv run mim_resnet.py --out runs/mim-resnet                  # full run with default settings
    uv run mim_resnet.py --smoke --out runs/mim-smoke           # 2 tiny epochs, checks the whole pipeline

Same idea as MAE (hide patches, reconstruct their pixels), adapted to a CNN, which cannot skip patches:
  1. The 96x96 image is split into a 12x12 grid of 8x8 patches; a random --mask-ratio of them is
     replaced by zeros (zero = the dataset mean colour after ImageNet normalisation).
  2. The ResNet9 (stable-pretraining's spt.backbone.Resnet9, as used for SwAV) processes the WHOLE
     masked image. Its last feature map before pooling is 256 x 12 x 12: one vector per 8x8 patch.
  3. A light decoder (a 1x1 conv by default, as in SimMIM) predicts the 8x8x3 pixels of every patch;
     the loss is the MSE on the masked patches only, with per-patch normalised targets (norm-pix).
  4. Exported embedding: the ResNet9 feature map, pooled (--pool). The ResNet9's final linear layer
     is never trained by this objective, so it is left out; a linear layer would not change a
     linear probe anyway.

BatchNorm sees masked images during training but full images at test time, so before every probe and
before export the BatchNorm statistics are recomputed on the unmasked, unaugmented training images.

Labels are used ONLY for the monitoring probe, never in the loss.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import stable_pretraining as spt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate import UNLABELLED, Ridge
from export_onnx import export_onnx
from mae_train import Augmented, load_images, lr_at, normalise

GRID, PATCH = 12, 8  # 96 = 12 * 8; matches the ResNet9's 12x12 feature map (stride 8)
POOLS = ("max2x2", "avg", "avg+max")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/train.parquet")
    p.add_argument("--out", default="runs/mim-resnet")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--crop-min", type=float, default=0.2)
    p.add_argument("--mask-ratio", type=float, default=0.6, help="SimMIM's default is 0.6; MAE uses 0.75")
    p.add_argument("--no-norm-pix", action="store_true")
    p.add_argument("--dec-depth", type=int, default=0, help="extra 3x3 conv layers before the 1x1 pixel head")
    p.add_argument("--dec-width", type=int, default=256)
    p.add_argument("--pool", choices=POOLS, default="max2x2",
                   help="max2x2 = the ResNet9's own pooling (1024 numbers); avg = global average (256)")
    p.add_argument("--probe-every", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--smoke", action="store_true", help="2 epochs on 512 images, to test the pipeline")
    return p.parse_args()


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

class ResNet9Encoder(nn.Module):
    """The ResNet9 convolutional body (everything but its final pooling and linear layer) + pooling."""

    def __init__(self, pool="max2x2"):
        super().__init__()
        resnet = spt.backbone.Resnet9(1024, 3)
        self.body = resnet.conv[:-1]  # drop AdaptiveMaxPool2d((2, 2)); output [B, 256, 12, 12]
        self.pool = pool

    def forward(self, x):
        f = self.body(x)
        if self.pool == "avg":
            return f.mean((2, 3))
        if self.pool == "max2x2":
            return F.adaptive_max_pool2d(f, 2).flatten(1)
        return torch.cat([f.mean((2, 3)), f.amax((2, 3))], dim=1)


class PixelDecoder(nn.Module):
    """Feature map [B, 256, 12, 12] -> image [B, 3, 96, 96]: optional 3x3 convs, then a 1x1 conv to
    the 8*8*3 pixels of each patch, rearranged into the image with pixel_shuffle."""

    def __init__(self, in_ch=256, depth=0, width=256):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU()]
            c = width
        layers.append(nn.Conv2d(c, PATCH * PATCH * 3, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return F.pixel_shuffle(self.net(f), PATCH)


def random_mask(batch, ratio, device):
    """[B, 1, 12, 12] with exactly round(ratio * 144) ones (1 = masked) per image."""
    n, k = GRID * GRID, int(round(GRID * GRID * ratio))
    ids = torch.rand(batch, n, device=device).argsort(1)
    mask = torch.zeros(batch, n, device=device).scatter_(1, ids[:, :k], 1.0)
    return mask.view(batch, 1, GRID, GRID)


def patchify(img):
    """[B, 3, 96, 96] -> [B, 144, 192]: the pixels of each 8x8 patch, row by row."""
    b = img.shape[0]
    return img.reshape(b, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, GRID * GRID, -1)


def mim_loss(pred, target, mask, norm_pix=True):
    pred, target = patchify(pred), patchify(target)
    if norm_pix:
        mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
    per_patch = ((pred - target) ** 2).mean(-1)  # [B, 144]
    m = mask.flatten(1)
    return (per_patch * m).sum() / m.sum()


# ---------------------------------------------------------------------------------------------------
# BatchNorm recalibration and monitoring probe
# ---------------------------------------------------------------------------------------------------

@torch.no_grad()
def recalibrate_bn(encoder, images, device, batch=256):
    """Recompute BatchNorm running statistics on full (unmasked) images, as the evaluator sees them."""
    bns = [m for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over all batches below
    was_training = encoder.training
    encoder.train()
    for i in range(0, len(images), batch):
        encoder.body(normalise(images[i:i + batch]).to(device))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    encoder.train(was_training)


@torch.no_grad()
def probe_all_pools(encoder, images, labels, device, batch=256):
    encoder.eval()
    f = torch.cat([encoder.body(normalise(images[i:i + batch]).to(device)).float().cpu()
                   for i in range(0, len(images), batch)])
    feats = {"max2x2": F.adaptive_max_pool2d(f, 2).flatten(1), "avg": f.mean((2, 3)),
             "avg+max": torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)}
    encoder.train()
    return {k: float((Ridge().fit(v.numpy().astype(np.float64), labels).loo_pred == labels).mean())
            for k, v in feats.items()}


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

    encoder = ResNet9Encoder(a.pool).to(device)
    decoder = PixelDecoder(depth=a.dec_depth, width=a.dec_width).to(device)
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"encoder {n_enc / 1e6:.2f}M params (~{n_enc * 4 / 2**20:.1f} MiB in ONNX), decoder {n_dec / 1e6:.2f}M")

    params = list(encoder.named_parameters()) + list(decoder.named_parameters())
    groups = [{"params": [p for _, p in params if p.ndim > 1], "weight_decay": a.weight_decay},
              {"params": [p for _, p in params if p.ndim <= 1], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = len(loader)
    total_steps, warmup_steps = a.epochs * steps_per_epoch, a.warmup_epochs * steps_per_epoch

    ckpt_path, start_epoch = os.path.join(a.out, "last.pt"), 0
    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ck["encoder"]); decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        print(f"resumed from {ckpt_path} at epoch {start_epoch}")
    else:
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(vars(a), f, indent=2)
        recalibrate_bn(encoder, images, device)
        acc0 = probe_all_pools(encoder, probe_images, probe_labels, device)
        print("epoch 0 (untrained) probe " + " ".join(f"{k} {v:.4f}" for k, v in acc0.items()))
        export_onnx(encoder, os.path.join(a.out, "model_init.onnx"))
        encoder.to(device)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    for epoch in range(start_epoch, a.epochs):
        encoder.train(); decoder.train()
        t0, loss_sum = time.time(), 0.0
        for i, x in enumerate(loader):
            lr = lr_at(epoch * steps_per_epoch + i, total_steps, warmup_steps, a.lr, a.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True)
            mask = random_mask(len(x), a.mask_ratio, device)
            pixel_mask = mask.repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)  # [B, 1, 96, 96]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = decoder(encoder.body(x * (1 - pixel_mask)))
            loss = mim_loss(pred.float(), x, mask, norm_pix=not a.no_norm_pix)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()

        rec = {"epoch": epoch + 1, "loss": loss_sum / steps_per_epoch, "lr": lr, "sec": round(time.time() - t0, 1)}
        if (epoch + 1) % a.probe_every == 0 or epoch + 1 == a.epochs:
            # Recalibrate a copy, so the training model keeps its own BatchNorm statistics
            snapshot = ResNet9Encoder(a.pool).to(device)
            snapshot.load_state_dict(encoder.state_dict())
            recalibrate_bn(snapshot, images, device)
            for k, v in probe_all_pools(snapshot, probe_images, probe_labels, device).items():
                rec[f"probe_{k}"] = v
        print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

        tmp = ckpt_path + ".tmp"
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "args": vars(a)}, tmp)
        os.replace(tmp, ckpt_path)

    recalibrate_bn(encoder, images, device)
    export_onnx(encoder, os.path.join(a.out, "model.onnx"))
    print(f"done: {a.out}/model.onnx")


if __name__ == "__main__":
    main()
