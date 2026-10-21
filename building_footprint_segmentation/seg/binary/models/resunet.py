"""A residual U-Net (ResUNet) for binary semantic segmentation.

Input:  float tensor [N, in_channels, H, W]
Output: logits      [N, num_classes, H, W]

The model returns logits (not probabilities), which is compatible with the
project's existing Dice/BCE criterion when num_classes=1.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Two-convolution residual block with a projection shortcut if needed."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ResidualBlock(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    """Upsample, concatenate the encoder skip and fuse residually."""

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.block = ResidualBlock(decoder_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(torch.cat((skip, x), dim=1))


class ResUNet(nn.Module):
    """Four-level U-Net using residual blocks in encoder and decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()
        if in_channels < 1 or num_classes < 1 or base_channels < 1:
            raise ValueError("Channel counts and base_channels must be positive")

        b = base_channels
        self.encoder1 = ResidualBlock(in_channels, b)
        self.encoder2 = DownBlock(b, 2 * b)
        self.encoder3 = DownBlock(2 * b, 4 * b)
        self.encoder4 = DownBlock(4 * b, 8 * b)
        self.bottleneck = DownBlock(8 * b, 16 * b)

        self.decoder4 = UpBlock(16 * b, 8 * b, 8 * b)
        self.decoder3 = UpBlock(8 * b, 4 * b, 4 * b)
        self.decoder2 = UpBlock(4 * b, 2 * b, 2 * b)
        self.decoder1 = UpBlock(2 * b, b, b)
        self.classifier = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)

        x = self.decoder4(b, e4)
        x = self.decoder3(x, e3)
        x = self.decoder2(x, e2)
        x = self.decoder1(x, e1)
        x = self.classifier(x)

        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x
