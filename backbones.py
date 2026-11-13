"""Small randomly initialized encoders for the 20 MiB image challenge."""

from torch import nn
from torchvision.models.resnet import BasicBlock


class CompactResNet(nn.Module):
    """BasicBlock ResNet with configurable width/depth and a small-image stem."""

    def __init__(self, width, blocks):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_channels = width
        for index, channels in enumerate((width, width * 2, width * 4, width * 8)):
            stride = 1 if index == 0 else 2
            downsample = None
            if stride != 1 or channels != in_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels),
                )
            layers = [BasicBlock(in_channels, channels, stride=stride, downsample=downsample)]
            layers.extend(BasicBlock(channels, channels) for _ in range(blocks[index] - 1))
            stages.append(nn.Sequential(*layers))
            in_channels = channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, image):
        return self.pool(self.stages(self.stem(image))).flatten(1)


class SmallResNet18(CompactResNet):
    """ResNet-18 with the existing width-32 or width-40 stage layout."""

    def __init__(self, width=32):
        super().__init__(width=width, blocks=(2, 2, 2, 2))


class SmallResNet34(CompactResNet):
    """ResNet-34 with width-28 stages to fit a single FP32 challenge export."""

    def __init__(self):
        super().__init__(width=28, blocks=(3, 4, 6, 3))
