"""Export modality-specific ONNX from existing weights (no training).

Usage:
  python -m ml.export.export_modality_onnx --modality fundus --weights ml/experiments/phase2_fundus/EfficientNetB0/best_weights.pt --model-name EfficientNetB0
  python -m ml.export.export_modality_onnx --modality anterior --weights ml/experiments/phase2_anterior/EfficientNetB0/best_weights.pt --model-name EfficientNetB0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.models.model_factory import create_model
from ml.export.export_onnx import export_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["fundus", "anterior"])
    parser.add_argument("--weights", required=True, help="Path to best_weights.pt")
    parser.add_argument("--model-name", default="EfficientNetB0")
    parser.add_argument("--out-dir", default="model_registry")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    weights_path = root / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{args.modality}_multi_disease.onnx"

    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    model = create_model(args.model_name, num_classes=4, pretrained=False)
    export_onnx(weights_path=str(weights_path), model=model, out_path=str(onnx_path))
    print("Exported", onnx_path)


if __name__ == "__main__":
    main()
