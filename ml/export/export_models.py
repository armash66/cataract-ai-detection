"""Unified export runner for EyeGPT-AI.
Exports two deployment targets:
- best-accuracy model
- lightweight model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.export.benchmark_export import benchmark_export_artifacts
from ml.export.export_onnx import export_onnx
from ml.export.export_tfjs import export_tfjs_from_onnx
from ml.export.quantize_model import quantize_onnx
from ml.models.model_factory import create_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-model", default="EfficientNetB0")
    parser.add_argument("--light-model", default="EyeGPTNet")
    parser.add_argument("--best-weights", default="")
    parser.add_argument("--light-weights", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    registry = Path("model_registry")
    registry.mkdir(parents=True, exist_ok=True)

    best_model = create_model(args.best_model, num_classes=4, pretrained=False)
    light_model = create_model(args.light_model, num_classes=4, pretrained=False)

    best_onnx = registry / "best_accuracy.onnx"
    light_onnx = registry / "lightweight.onnx"

    export_onnx(weights_path=args.best_weights, model=best_model, out_path=str(best_onnx))
    export_onnx(weights_path=args.light_weights, model=light_model, out_path=str(light_onnx))

    best_q = registry / "best_accuracy.int8.onnx"
    light_q = registry / "lightweight.int8.onnx"
    quantize_onnx(str(best_onnx), str(best_q))
    quantize_onnx(str(light_onnx), str(light_q))

    export_tfjs_from_onnx(str(best_onnx), str(registry / "tfjs_best"))

    benchmark_export_artifacts([
        str(best_onnx),
        str(best_q),
        str(light_onnx),
        str(light_q),
    ], out_file=str(registry / "model_performance_benchmark.json"))

    metadata = {
        "best_accuracy_model": args.best_model,
        "lightweight_model": args.light_model,
        "artifacts": {
            "best_onnx": str(best_onnx),
            "best_quantized": str(best_q),
            "light_onnx": str(light_onnx),
            "light_quantized": str(light_q),
        },
    }
    (registry / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
