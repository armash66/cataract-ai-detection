"""Model factory for baseline and advanced ophthalmology classifiers."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

MODEL_NAMES = [
    "EfficientNetB0",
    "ResNet50",
    "MobileNetV3",
    "EfficientNetV2",
    "ViT",
    "ConvNeXt",
    "CNNAttentionHybrid",
]


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.proj(self.pool(x))
        return x * w


class CNNAttentionHybrid(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            AttentionBlock(128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


def _replace_linear(module: nn.Module, attr_name: str, num_classes: int) -> None:
    layer = getattr(module, attr_name)
    if isinstance(layer, nn.Linear):
        setattr(module, attr_name, nn.Linear(layer.in_features, num_classes))
    else:
        raise TypeError(f"Unsupported classifier type for {attr_name}: {type(layer)}")


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        _replace_linear(model.classifier, "1", num_classes)
        return model

    if name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        _replace_linear(model, "fc", num_classes)
        return model

    if name == "MobileNetV3":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        _replace_linear(model.classifier, "3", num_classes)
        return model

    if name == "EfficientNetV2":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        _replace_linear(model.classifier, "1", num_classes)
        return model

    if name == "ViT":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        _replace_linear(model.heads, "head", num_classes)
        return model

    if name == "ConvNeXt":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        _replace_linear(model.classifier, "2", num_classes)
        return model

    if name == "CNNAttentionHybrid":
        return CNNAttentionHybrid(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "ResNet50":
        for p in model.fc.parameters():
            p.requires_grad = True
    elif model_name == "ViT":
        for p in model.heads.parameters():
            p.requires_grad = True
    elif model_name == "CNNAttentionHybrid":
        for p in model.parameters():
            p.requires_grad = True
    else:
        classifier = getattr(model, "classifier", None)
        if classifier is not None:
            for p in classifier.parameters():
                p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
