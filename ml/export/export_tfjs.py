"""TensorFlow.js export hook.
Requires onnx-tf + tensorflowjs_converter in environment.
"""
from __future__ import annotations

from pathlib import Path


def export_tfjs_from_onnx(onnx_path: str, out_dir: str = "model_registry/tfjs"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("TODO: convert", onnx_path, "to TF.js in", out)
