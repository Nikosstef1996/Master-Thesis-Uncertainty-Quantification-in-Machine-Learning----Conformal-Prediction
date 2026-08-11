from __future__ import annotations

import torch
from torch import nn
from torchvision import models


SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        metadata_dim: int,
        pretrained: bool = True,
        metadata_hidden: int = 64,
        classifier_hidden: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.metadata_dim = metadata_dim
        self.image_encoder, image_dim = build_image_encoder(backbone, pretrained=pretrained)

        if metadata_dim > 0:
            self.metadata_encoder = nn.Sequential(
                nn.LayerNorm(metadata_dim),
                nn.Linear(metadata_dim, metadata_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fused_dim = image_dim + metadata_hidden
        else:
            self.metadata_encoder = None
            fused_dim = image_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        if self.metadata_encoder is not None:
            metadata_features = self.metadata_encoder(metadata)
            features = torch.cat([image_features, metadata_features], dim=1)
        else:
            features = image_features
        return self.classifier(features)

    def freeze_image_encoder(self) -> None:
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False


def build_image_encoder(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone {backbone!r}. Choose from {SUPPORTED_BACKBONES}.")

    if backbone == "resnet18":
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            model = models.resnet18(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    if backbone == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        except (AttributeError, TypeError):
            model = models.resnet50(pretrained=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    except (AttributeError, TypeError):
        model = models.efficientnet_b0(pretrained=pretrained)
    feature_dim = model.classifier[-1].in_features
    model.classifier = nn.Identity()
    return model, feature_dim
