"""Local timm bridge; no installed package modifications or pretrained weights."""
import os
from pathlib import Path

# SPT has its own cache setting independent of XDG_CACHE_HOME.
os.environ.setdefault("SPT_CACHE_DIR", str(Path(__file__).resolve().parents[2] /
                                         "experiment_outputs/cache/stable-pretraining"))

import torch
from timm.models import register_model, is_model
from stable_pretraining.backbone import Resnet9
from stable_pretraining.methods import LeJEPA
from .readouts import SpatialReadout, output_dim
from .resnet18 import CompactResNet18


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
    spatial_channels = 256

    def __init__(self, readout='max2'):
        super().__init__(num_classes=1, num_channels=3)
        self.fc = torch.nn.Identity()
        self.conv[-1] = DeterministicAdaptiveMaxPool2d()
        self.readout_kind = readout
        self.readout = SpatialReadout(readout)
        self.embed_dim = output_dim(256, readout)

    def aggregate(self, features):
        return self.readout(features)

    def forward(self, image):
        if self.readout_kind == 'max2':
            return super().forward(image)
        return self.aggregate(self.spatial(image))

    def spatial(self, x):
        for layer in list(self.conv.children())[:-1]:
            x = layer(x)
        return x


@register_model
def challenge_resnet9(pretrained=False, num_classes=0, **kwargs):
    if pretrained or num_classes != 0:
        raise ValueError("Only randomly initialized encoders are supported")
    return Encoder()


def make_encoder(config=None):
    c = config or {}
    backbone = c.get('backbone', 'resnet9')
    if backbone == 'resnet9':
        return Encoder(c.get('readout', 'max2'))
    if backbone == 'resnet18':
        return CompactResNet18(c.get('stage_widths', [40,80,160,320]),
                               c.get('stage_depths', [2,2,2,2]), c.get('readout', 'gap'))
    raise ValueError(f'Unsupported backbone: {backbone}')


def make_method(config):
    name = 'challenge_resnet9'
    if config.get('backbone', 'resnet9') != 'resnet9' or config.get('readout', 'max2') != 'max2':
        name = 'challenge_' + config.get('backbone', 'resnet9') + '_' + config.get('readout', 'gap')
        if config.get('backbone') == 'resnet18':
            name += '_' + '_'.join(map(str, config.get('stage_widths', [40,80,160,320])))
        if not is_model(name):
            def factory(pretrained=False, num_classes=0, **kwargs):
                if pretrained or num_classes != 0:
                    raise ValueError('Only randomly initialized feature encoders supported')
                return make_encoder(config)
            factory.__name__ = name
            register_model(factory)
    return LeJEPA(encoder_name=name, pretrained=False,
                  lamb=config["lamb"], n_slices=config["n_slices"], drop_path_rate=0.)


def load_encoder(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get('hyper_parameters', {}).get('config', {})
    encoder = make_encoder(config)
    prefix = "method.backbone."
    encoder.load_state_dict({k[len(prefix):]: v for k, v in checkpoint["state_dict"].items()
                             if k.startswith(prefix)}, strict=True)
    encoder.experiment_config = config
    return encoder.eval()
