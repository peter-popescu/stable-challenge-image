"""Shared EMA encoder export for periodic scoring and final submission."""

import copy
from pathlib import Path

import torch

from export_onnx import export_onnx
from backbones import CompactResNet


class MultiScaleResNet(torch.nn.Module):
    """Concatenate average-pooled stages 2, 3, and 4 of one existing encoder."""

    def __init__(self, encoder):
        super().__init__()
        if not isinstance(encoder, CompactResNet):
            raise ValueError("Multiscale export requires a compact ResNet backbone")
        self.encoder = encoder

    def forward(self, image):
        features = []
        x = self.encoder.stem(image)
        for index, stage in enumerate(self.encoder.stages):
            x = stage(x)
            if index >= 1:
                features.append(self.encoder.pool(x).flatten(1))
        return torch.cat(features, dim=1)


def export_teacher(model, path, features="final"):
    encoder = copy.deepcopy(model.backbone.teacher).cpu().eval()
    if features == "multiscale":
        encoder = MultiScaleResNet(encoder).eval()
    elif features != "final":
        raise ValueError(f"Unknown export features: {features}")
    if hasattr(encoder, "patch_embed"):
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
    print(f"EMA teacher encoder ({features} features): {size:.2f} MiB. Validate with test_submission.py.")
    return path
