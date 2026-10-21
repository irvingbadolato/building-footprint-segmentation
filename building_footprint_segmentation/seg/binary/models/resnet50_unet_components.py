"""Shared building blocks for U-Net decoders using a ResNet-50 encoder."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Encoder(nn.Module):
    """ResNet-50 feature encoder exposing stride 2, 4, 8, 16 and 32 maps.

    The ImageNet V1 weights are selected explicitly so this encoder can start
    from the same ResNet-50 weight set as ``ReFineNet(pre_trained_image_net=True)``
    when it uses torchvision's legacy ``pretrained=True`` argument.

    Args:
        in_channels: Number of input image bands/channels.
        pre_trained_image_net: Initialize from ImageNet-1K V1 weights.
        trainable: If False, freeze parameters and keep BatchNorm layers in eval.
    """

    # Feature channels returned by forward(): stem, layer1, layer2, layer3, layer4.
    out_channels = (64, 256, 512, 1024, 2048)

    def __init__(
        self,
        in_channels: int = 3,
        pre_trained_image_net: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        weights = (
            ResNet50_Weights.IMAGENET1K_V1
            if pre_trained_image_net
            else None
        )
        backbone = resnet50(weights=weights)

        if in_channels != 3:
            backbone.conv1 = self._adapt_first_convolution(
                backbone.conv1,
                in_channels=in_channels,
                pretrained=pre_trained_image_net,
            )

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.trainable = bool(trainable)

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False
            # Keep BN statistics fixed when the encoder is frozen.
            self.eval()

    @staticmethod
    def _adapt_first_convolution(
        original: nn.Conv2d,
        in_channels: int,
        pretrained: bool,
    ) -> nn.Conv2d:
        """Adapt conv1 for non-RGB inputs while preserving pretrained weights.

        For 1--2 channels, the available RGB weights are averaged/subselected
        and rescaled. For more than 3 channels, RGB kernels are repeated and
        scaled to avoid increasing the initial response solely because there
        are more input bands. This is an initialization heuristic, not a
        substitute for validating multispectral preprocessing.
        """
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            dilation=original.dilation,
            groups=original.groups,
            bias=original.bias is not None,
            padding_mode=original.padding_mode,
        )

        if pretrained:
            old_weight = original.weight.detach()
            with torch.no_grad():
                if in_channels == 1:
                    adapted = old_weight.mean(dim=1, keepdim=True)
                elif in_channels < 3:
                    adapted = old_weight[:, :in_channels] * (3.0 / in_channels)
                else:
                    repeats = ceil(in_channels / 3)
                    adapted = old_weight.repeat(1, repeats, 1, 1)[:, :in_channels]
                    adapted = adapted * (3.0 / in_channels)
                new_conv.weight.copy_(adapted)
                if new_conv.bias is not None and original.bias is not None:
                    new_conv.bias.copy_(original.bias.detach())
        return new_conv

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        stem = self.stem(x)                       # 1/2 resolution
        layer1 = self.layer1(self.maxpool(stem))  # 1/4
        layer2 = self.layer2(layer1)              # 1/8
        layer3 = self.layer3(layer2)              # 1/16
        layer4 = self.layer4(layer3)              # 1/32
        return stem, layer1, layer2, layer3, layer4

    def train(self, mode: bool = True) -> "ResNet50Encoder":
        """Keep the frozen encoder (including BatchNorm) in evaluation mode."""
        super().train(mode)
        if not self.trainable:
            self.stem.eval()
            self.maxpool.eval()
            self.layer1.eval()
            self.layer2.eval()
            self.layer3.eval()
            self.layer4.eval()
        return self


class DoubleConv(nn.Module):
    """Two Conv-BatchNorm-ReLU layers used by the conventional U-Net decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ResidualConvBlock(nn.Module):
    """Two-convolution residual block with a projection shortcut if needed."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
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
            self.shortcut: nn.Module = nn.Identity()
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


class UNetDecoderBlock(nn.Module):
    """Upsample, concatenate the encoder skip, and fuse with DoubleConv."""

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.fuse = DoubleConv(decoder_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat((skip, x), dim=1))


class ResUNetDecoderBlock(nn.Module):
    """Upsample and concatenate, then use channel projection + residual block.

    The 1x1 projection limits decoder parameter count when the deepest ResNet
    features (2048 channels) are concatenated with a large skip tensor.
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(
                decoder_channels + skip_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualConvBlock(out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.fuse(torch.cat((skip, x), dim=1))
        return self.residual(x)


def validate_decoder_channels(channels: Sequence[int]) -> tuple[int, int, int, int]:
    channels = tuple(int(c) for c in channels)
    if len(channels) != 4 or any(c < 1 for c in channels):
        raise ValueError(
            "decoder_channels must contain four positive integers, "
            "for example (256, 128, 64, 32)"
        )
    return channels
