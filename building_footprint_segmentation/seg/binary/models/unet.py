"""A compact U-Net for binary semantic segmentation.

Input:  float tensor [N, in_channels, H, W]
Output: logits      [N, num_classes, H, W]

For the existing binary Dice criterion, use num_classes=1 and apply the
sigmoid in the criterion/metric rather than inside the model.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample to the skip's exact shape, concatenate, then convolve."""

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Four-level U-Net with configurable input/output channels and width."""

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
        self.encoder1 = DoubleConv(in_channels, b)
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

        # Defensive: retain exactly the input spatial dimensions.
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x
