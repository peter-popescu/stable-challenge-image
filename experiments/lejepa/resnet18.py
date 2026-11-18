"""Small-image BasicBlock ResNet matching the pinned DINO teammate backbone."""
from torch import nn
from torchvision.models.resnet import BasicBlock
from .readouts import SpatialReadout, output_dim


class CompactResNet18(nn.Module):
    def __init__(self, widths=(40, 80, 160, 320), depths=(2, 2, 2, 2), readout='gap'):
        super().__init__()
        if len(widths) != 4 or tuple(depths) != (2, 2, 2, 2) or any(type(w) is not int or w < 1 for w in widths):
            raise ValueError('Require four positive stage widths and ResNet18 depths [2,2,2,2]')
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages, incoming = [], widths[0]
        for index, channels in enumerate(widths):
            stride = 1 if index == 0 else 2
            downsample = None
            if stride != 1 or channels != incoming:
                downsample = nn.Sequential(nn.Conv2d(incoming, channels, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(channels))
            layers = [BasicBlock(incoming, channels, stride=stride, downsample=downsample)]
            layers.extend(BasicBlock(channels, channels) for _ in range(depths[index]-1))
            stages.append(nn.Sequential(*layers))
            incoming = channels
        self.stages = nn.Sequential(*stages)
        self.pool = SpatialReadout(readout)
        self.spatial_channels = widths[-1]
        self.embed_dim = output_dim(widths[-1], readout)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def spatial(self, image):
        return self.stages(self.stem(image))

    def aggregate(self, features):
        return self.pool(features)

    def forward(self, image):
        return self.aggregate(self.spatial(image))
