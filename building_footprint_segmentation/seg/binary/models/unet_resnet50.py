"""U-Net with an ImageNet-pretrained ResNet-50 encoder.

Input:  float tensor [N, in_channels, H, W]
Output: logits tensor [N, num_classes, H, W] (no sigmoid in the model).
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn
from torch.nn import functional as F

from .resnet50_unet_components import (
    ResNet50Encoder,
    UNetDecoderBlock,
    validate_decoder_channels,
)


class UNetResNet50(nn.Module):
    """U-Net decoder with a ResNet-50 feature encoder.

    Args:
        in_channels: Number of raster bands. RGB is the recommended baseline.
        num_classes: Use 1 for the repository's binary Dice/BCE criterion.
        pre_trained_image_net: Load torchvision ResNet-50 ImageNet V1 weights.
        top_layers_trainable: Retains RefineNet's argument name; controls whether
            encoder parameters can be fine-tuned. If False, encoder BatchNorm
            statistics are also frozen.
        decoder_channels: Widths of decoder levels, shallow to deep in order
            of construction: (d4, d3, d2, d1), default (256, 128, 64, 32).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        pre_trained_image_net: bool = True,
        top_layers_trainable: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive")

        d4, d3, d2, d1 = validate_decoder_channels(decoder_channels)
        self.num_classes = num_classes
        self.encoder = ResNet50Encoder(
            in_channels=in_channels,
            pre_trained_image_net=pre_trained_image_net,
            trainable=top_layers_trainable,
        )

        # ResNet-50 feature channels: stem=64, layer1=256, layer2=512,
        # layer3=1024 and layer4=2048.
        self.decoder4 = UNetDecoderBlock(2048, 1024, d4)
        self.decoder3 = UNetDecoderBlock(d4, 512, d3)
        self.decoder2 = UNetDecoderBlock(d3, 256, d2)
        self.decoder1 = UNetDecoderBlock(d2, 64, d1)
        self.classifier = nn.Conv2d(d1, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        stem, enc1, enc2, enc3, enc4 = self.encoder(x)

        x = self.decoder4(enc4, enc3)
        x = self.decoder3(x, enc2)
        x = self.decoder2(x, enc1)
        x = self.decoder1(x, stem)
        x = F.interpolate(
            x, size=output_size, mode="bilinear", align_corners=False
        )
        return self.classifier(x)
