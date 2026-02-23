from __future__ import annotations

from pathlib import Path


def quantize_onnx(onnx_in: str, onnx_out: str = "model_registry/best_model.int8.onnx"):
    # Placeholder to keep structure stable. Replace with onnxruntime.quantization in full env.
    src = Path(onnx_in)
    dst = Path(onnx_out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dst.write_bytes(src.read_bytes())
    print("Saved", dst)
