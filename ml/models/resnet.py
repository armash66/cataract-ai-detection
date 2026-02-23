from torchvision import models
from ml.models._common import replace_linear


def build_resnet50(num_classes: int, pretrained: bool = True):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    replace_linear(model, "fc", num_classes)
    return model
