from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(weights_path: str, model, out_path: str = "model_registry/best_model.onnx"):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if weights_path and Path(weights_path).exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print("Saved", out)
