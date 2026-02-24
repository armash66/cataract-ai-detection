"""
Confidence calibration and per-class threshold tuning.

- Temperature scaling: fit a single scalar T on validation logits so softmax(logits/T) is better calibrated.
- Per-class thresholds: optional thresholds for abstention or binary decisions per class (tuned on val set).

Usage:
  python -m ml.training.calibrate_and_thresholds --data-dir ml/experiments/phase1/cleaned \\
    --weights ml/experiments/phase2/EfficientNetB0/best_weights.pt --model-name EfficientNetB0 \\
    --out model_registry/calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import brier_score_loss

from ml.models.model_factory import create_model
from ml.training.trainer_utils import get_dataloaders

CANONICAL_CLASSES = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]


def softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """logits shape (N, C), return (N, C) probabilities."""
    x = logits / max(temperature, 1e-6)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def nll_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """Average NLL for multiclass."""
    n = probs.shape[0]
    return -np.log(probs[np.arange(n), labels] + 1e-8).mean()


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T that minimizes NLL(softmax(logits/T), labels)."""
    def obj(T: float) -> float:
        p = softmax_temperature(logits, T)
        return nll_loss(p, labels)

    res = minimize_scalar(obj, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def per_class_thresholds_from_val(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    num_steps: int = 50,
) -> dict[str, float]:
    """
    For each class, find threshold that maximizes Youden index (sensitivity + specificity - 1)
    on one-vs-rest binary view. Returns recommended abstain threshold per class (below = uncertain).
    """
    thresholds = {}
    for c in range(len(class_names)):
        binary = (y_true == c).astype(np.int32)
        prob_c = y_prob[:, c]
        best_j = -1.0
        best_t = 0.5
        for t in np.linspace(0.1, 0.9, num_steps):
            pred = (prob_c >= t).astype(np.int32)
            tp = ((pred == 1) & (binary == 1)).sum()
            tn = ((pred == 0) & (binary == 0)).sum()
            fp = ((pred == 1) & (binary == 0)).sum()
            fn = ((pred == 0) & (binary == 1)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            youden = sens + spec - 1.0
            if youden > best_j:
                best_j = youden
                best_t = t
        thresholds[class_names[c]] = round(best_t, 4)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate model and compute per-class thresholds.")
    parser.add_argument("--data-dir", default="ml/experiments/phase1/cleaned", help="Path to cleaned train/val/test")
    parser.add_argument("--weights", required=True, help="Path to best_weights.pt")
    parser.add_argument("--model-name", default="EfficientNetB0", help="Model architecture name")
    parser.add_argument("--out", default="model_registry/calibration.json", help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_dir = root / args.data_dir
    weights_path = root / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    out_path = root / args.out if not Path(args.out).is_absolute() else Path(args.out)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    _, val_loader, _, classes = get_dataloaders(data_dir, args.batch_size, args.num_workers)
    num_classes = len(classes)
    model = create_model(args.model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=args.device))
    model = model.to(args.device)

    # Collect raw logits on val set (we need logits for temperature scaling)
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(args.device)
            logits = model(x)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Fit temperature
    temperature = fit_temperature(logits, labels)
    probs_calibrated = softmax_temperature(logits, temperature)

    # Brier score before/after (lower is better)
    probs_raw = softmax_temperature(logits, 1.0)
    brier_raw = brier_score_loss(labels, probs_raw, multi_class="ovr")
    brier_cal = brier_score_loss(labels, probs_calibrated, multi_class="ovr")

    # Per-class thresholds (for abstention / per-class decision)
    class_names = list(classes)
    per_class_thresholds = per_class_thresholds_from_val(labels, probs_calibrated, class_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "temperature": round(temperature, 6),
        "per_class_thresholds": per_class_thresholds,
        "global_abstain_threshold": 0.5,
        "brier_raw": round(float(brier_raw), 6),
        "brier_calibrated": round(float(brier_cal), 6),
        "model_name": args.model_name,
        "weights_path": str(weights_path),
        "class_names": class_names,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved calibration to {out_path}")
    print(f"  temperature = {temperature:.4f}")
    print(f"  Brier (raw) = {brier_raw:.4f}, Brier (calibrated) = {brier_cal:.4f}")
    print("  per_class_thresholds:", per_class_thresholds)


if __name__ == "__main__":
    main()
