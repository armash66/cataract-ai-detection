"""Unified export runner for EyeGPT-AI."""
from __future__ import annotations

from ml.export.benchmark_export import benchmark_export_artifacts
from ml.export.export_onnx import export_onnx
from ml.export.export_tfjs import export_tfjs_from_onnx
from ml.export.quantize_model import quantize_onnx
from ml.models.model_factory import create_model


def main():
    model = create_model("EyeGPTNet", num_classes=4, pretrained=False)
    export_onnx(weights_path="", model=model, out_path="model_registry/best_model.onnx")
    quantize_onnx("model_registry/best_model.onnx", "model_registry/best_model.int8.onnx")
    export_tfjs_from_onnx("model_registry/best_model.onnx", "model_registry/tfjs")
    benchmark_export_artifacts([
        "model_registry/best_model.onnx",
        "model_registry/best_model.int8.onnx",
    ])


if __name__ == "__main__":
    main()
