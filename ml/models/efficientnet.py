from torchvision import models
from ml.models._common import replace_linear


def build_efficientnet_b0(num_classes: int, pretrained: bool = True):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    replace_linear(model.classifier, "1", num_classes)
    return model


def build_efficientnet_v2(num_classes: int, pretrained: bool = True):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
    replace_linear(model.classifier, "1", num_classes)
    return model
