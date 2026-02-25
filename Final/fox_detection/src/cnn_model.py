"""
cnn_model.py
CNN architectures for binary fox / non-fox spectrogram classification.
Supports EfficientNet-B0, ResNet-18, and MobileNet-V3-Small backbones
from torchvision, with a custom classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class FoxCNN(nn.Module):
    """Transfer-learning CNN for fox audio detection.

    Parameters
    ----------
    backbone : str
        One of ``'efficientnet_b0'``, ``'resnet18'``, ``'mobilenet_v3_small'``.
    pretrained : bool
        If ``True``, load ImageNet pre-trained weights (default ``True``).
    num_classes : int
        Number of output classes (default 2).
    dropout : float
        Dropout probability before the final linear layer (default 0.3).
    """

    _SUPPORTED_BACKBONES = {"efficientnet_b0", "resnet18", "mobilenet_v3_small"}

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone not in self._SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone {backbone!r}. "
                f"Choose from {sorted(self._SUPPORTED_BACKBONES)}"
            )

        self.backbone_name = backbone
        weights = "DEFAULT" if pretrained else None

        # ── Load backbone & replace classifier head ───────────────────
        if backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone == "resnet18":
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=weights)
            in_features = base.classifier[-1].in_features
            base.classifier[-1] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        return self.model(x)
