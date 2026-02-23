from torch import nn


def replace_linear(module: nn.Module, attr_name: str, num_classes: int) -> None:
    layer = getattr(module, attr_name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Unsupported classifier type for {attr_name}: {type(layer)}")
    setattr(module, attr_name, nn.Linear(layer.in_features, num_classes))
