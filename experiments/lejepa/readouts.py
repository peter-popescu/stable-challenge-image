"""Parameter-free spatial aggregation shared by training and frozen evaluation."""
import torch
from torch import nn

READOUTS = ('max2', 'avg2', 'gap', 'gmp', 'avgmax', 'avgmaxstd')


def output_dim(channels, readout):
    factors = dict(max2=4, avg2=4, gap=1, gmp=1, avgmax=2, avgmaxstd=3)
    return channels * factors[readout]


class SpatialReadout(nn.Module):
    def __init__(self, kind='max2'):
        super().__init__()
        if kind not in READOUTS:
            raise ValueError(f'Unknown readout: {kind}')
        self.kind = kind

    def forward(self, x):
        if self.kind == 'max2':
            # Identical adaptive bins and first-max tie handling to our encoder.
            h, w = x.shape[-2:]
            bins = [x[:, :, i*h//2:((i+1)*h+1)//2, j*w//2:((j+1)*w+1)//2]
                    .flatten(2).max(-1).values for i in range(2) for j in range(2)]
            return torch.stack(bins, -1).flatten(1)
        if self.kind == 'avg2':
            return torch.nn.functional.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
        avg = x.mean((-2, -1))
        if self.kind == 'gap':
            return avg
        maximum = x.flatten(2).max(-1).values
        if self.kind == 'gmp':
            return maximum
        if self.kind == 'avgmax':
            return torch.cat([avg, maximum], 1)
        # Native std uses a zero subgradient at zero variance (unlike sqrt(var)).
        # Keep population convention and exact frozen values; no epsilon bias.
        return torch.cat([avg, maximum, x.std((-2, -1), unbiased=False)], 1)


class FrozenReadout(nn.Module):
    def __init__(self, encoder, kind):
        super().__init__()
        self.encoder = encoder.eval()
        self.pool = SpatialReadout(kind)
        self.embed_dim = output_dim(encoder.spatial_channels, kind)
        self.requires_grad_(False)

    def forward(self, x):
        return self.pool(self.encoder.spatial(x))
