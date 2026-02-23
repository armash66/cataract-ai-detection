from torchvision import models
from ml.models._common import replace_linear


def build_vit_b16(num_classes: int, pretrained: bool = True):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
    replace_linear(model.heads, "head", num_classes)
    return model
