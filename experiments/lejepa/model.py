"""Local timm bridge; no installed package modifications or pretrained weights."""
import os
from pathlib import Path

# SPT has its own cache setting independent of XDG_CACHE_HOME.
os.environ.setdefault("SPT_CACHE_DIR", str(Path(__file__).resolve().parents[2] /
                                         "experiment_outputs/cache/stable-pretraining"))

import torch
from timm.models import register_model
from stable_pretraining.backbone import Resnet9
from stable_pretraining.methods import LeJEPA


class DeterministicAdaptiveMaxPool2d(torch.nn.Module):
    """Exact adaptive 2x2 bins with first-maximum tie handling.

    CUDA adaptive_max_pool2d backward rejects strict determinism. Reducing
    each bin with indexed max preserves its values and selected gradients,
    including overlapping bins for odd spatial dimensions.
    """

    def forward(self, x):
        height, width = x.shape[-2:]
        bins = []
        for row in range(2):
            for col in range(2):
                tile = x[:, :, row * height // 2:((row + 1) * height + 1) // 2,
                         col * width // 2:((col + 1) * width + 1) // 2]
                bins.append(tile.flatten(2).max(dim=-1).values)
        return torch.stack(bins, dim=-1).reshape(x.shape[0], x.shape[1], 2, 2)


class Encoder(Resnet9):
    embed_dim = 1024

    def __init__(self):
        super().__init__(num_classes=1, num_channels=3)
        self.fc = torch.nn.Identity()
        self.conv[-1] = DeterministicAdaptiveMaxPool2d()

    def spatial(self, x):
        for layer in list(self.conv.children())[:-1]:
            x = layer(x)
        return x


@register_model
def challenge_resnet9(pretrained=False, num_classes=0, **kwargs):
    if pretrained or num_classes != 0:
        raise ValueError("Only randomly initialized encoders are supported")
    return Encoder()


def make_method(config):
    return LeJEPA(encoder_name="challenge_resnet9", pretrained=False,
                  lamb=config["lamb"], n_slices=config["n_slices"], drop_path_rate=0.)


def load_encoder(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = Encoder()
    prefix = "method.backbone."
    encoder.load_state_dict({k[len(prefix):]: v for k, v in checkpoint["state_dict"].items()
                             if k.startswith(prefix)}, strict=True)
    return encoder.eval()
