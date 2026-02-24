"""Train a single modality-specific model (fundus or anterior) and export to ONNX.

Usage:
  python -m ml.training.train_modality --modality fundus --data-dir ml/experiments/phase1_fundus/cleaned --output-dir ml/experiments/phase2_fundus
  python -m ml.training.train_modality --modality anterior --data-dir ml/experiments/phase1_anterior/cleaned --output-dir ml/experiments/phase2_anterior

Exports to model_registry/{modality}_multi_disease.onnx when training finishes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.models.model_factory import create_model
from ml.export.export_onnx import export_onnx
from ml.training import train_and_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["fundus", "anterior"])
    parser.add_argument("--data-dir", default="ml/experiments/phase1/cleaned")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="EfficientNetB0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-freeze", type=int, default=2)
    parser.add_argument("--epochs-finetune", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    output_dir = ROOT / (args.output_dir or f"ml/experiments/phase2_{args.modality}")
    registry = ROOT / "model_registry"
    registry.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise FileNotFoundError(f"Expected train/val under {data_dir}")

    # Invoke train_and_benchmark with modality-specific dirs
    old_argv = sys.argv
    sys.argv = [
        "train_and_benchmark",
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--models", args.model,
        "--batch-size", str(args.batch_size),
        "--epochs-freeze", str(args.epochs_freeze),
        "--epochs-finetune", str(args.epochs_finetune),
        "--device", args.device,
    ]
    try:
        train_and_benchmark.main()
    finally:
        sys.argv = old_argv

    # Export best weights to modality-specific ONNX
    best_weights = output_dir / args.model / "best_weights.pt"
    if not best_weights.exists():
        print("No best_weights.pt found; skipping ONNX export.")
        return
    onnx_path = registry / f"{args.modality}_multi_disease.onnx"
    model = create_model(args.model, num_classes=4, pretrained=False)
    export_onnx(weights_path=str(best_weights), model=model, out_path=str(onnx_path))
    print("Exported", onnx_path)


if __name__ == "__main__":
    main()
