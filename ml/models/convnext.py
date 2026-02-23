from torchvision import models
from ml.models._common import replace_linear


def build_convnext_tiny(num_classes: int, pretrained: bool = True):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
    replace_linear(model.classifier, "2", num_classes)
    return model
