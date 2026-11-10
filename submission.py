"""Shared EMA encoder export for periodic scoring and final submission."""

import copy
from pathlib import Path

import torch

from export_onnx import export_onnx


def export_teacher(model, path):
    encoder = copy.deepcopy(model.backbone.teacher).cpu().eval()
    encoder.dynamic_img_size = False
    encoder.patch_embed.strict_img_size = True
    encoder.patch_embed.output_fmt = "NCHW"
    encoder.patch_embed.flatten = True
    for block in encoder.blocks:
        block.attn.fused_attn = False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tracing's random example input must not change the training RNG stream.
    with torch.random.fork_rng(devices=[]):
        export_onnx(encoder, str(path))
    size = path.stat().st_size / 1024**2
    if size > 20:
        raise ValueError(f"ONNX file is {size:.2f} MiB; challenge limit is 20 MiB")
    print(f"EMA teacher encoder: {size:.2f} MiB. Validate with test_submission.py.")
    return path
