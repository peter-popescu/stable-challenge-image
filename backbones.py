"""Small randomly initialized encoders for the 20 MiB image challenge."""

from torch import nn
from torchvision.models.resnet import BasicBlock


class SmallResNet18(nn.Module):
    """ResNet-18 block layout with half-width stages and a low-resolution stem."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_channels = 32
        for index, channels in enumerate((32, 64, 128, 256)):
            stride = 1 if index == 0 else 2
            downsample = None
            if stride != 1 or channels != in_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(channels),
                )
            stages.append(nn.Sequential(
                BasicBlock(in_channels, channels, stride=stride, downsample=downsample),
                BasicBlock(channels, channels),
            ))
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
