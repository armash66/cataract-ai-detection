"""Central model factory."""
from __future__ import annotations

from torch import nn

from ml.models.attention_cnn import CNNAttentionHybrid
from ml.models.convnext import build_convnext_tiny
from ml.models.efficientnet import build_efficientnet_b0, build_efficientnet_v2
from ml.models.eyegptnet import EyeGPTNet
from ml.models.mobilenet import build_mobilenet_v3
from ml.models.resnet import build_resnet50
from ml.models.vit import build_vit_b16

MODEL_NAMES = [
    "EfficientNetB0",
    "ResNet50",
    "MobileNetV3",
    "EfficientNetV2",
    "ViT",
    "ConvNeXt",
    "CNNAttentionHybrid",
    "EyeGPTNet",
]


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "EfficientNetB0":
        return build_efficientnet_b0(num_classes, pretrained)
    if name == "ResNet50":
        return build_resnet50(num_classes, pretrained)
    if name == "MobileNetV3":
        return build_mobilenet_v3(num_classes, pretrained)
    if name == "EfficientNetV2":
        return build_efficientnet_v2(num_classes, pretrained)
    if name == "ViT":
        return build_vit_b16(num_classes, pretrained)
    if name == "ConvNeXt":
        return build_convnext_tiny(num_classes, pretrained)
    if name == "CNNAttentionHybrid":
        return CNNAttentionHybrid(num_classes)
    if name == "EyeGPTNet":
        return EyeGPTNet(num_classes=num_classes)
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
    elif model_name in {"CNNAttentionHybrid", "EyeGPTNet"}:
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
