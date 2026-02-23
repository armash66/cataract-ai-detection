from torchvision import models
from ml.models._common import replace_linear


def build_mobilenet_v3(num_classes: int, pretrained: bool = True):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    replace_linear(model.classifier, "3", num_classes)
    return model
