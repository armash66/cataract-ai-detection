"""External validation: evaluate on a held-out source/domain and report internal vs external gap.

Usage:
  python -m ml.evaluation.external_validation \\
    --internal-metrics ml/experiments/phase2/EfficientNetB0/metrics.json \\
    --external-csv ml/experiments/external_test_labels.csv \\
    --weights ml/experiments/phase2/EfficientNetB0/best_weights.pt \\
    --model-name EfficientNetB0 \\
    --out ml/experiments/external_validation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]


def load_model(weights_path: Path, model_name: str, num_classes: int = 4):
    from ml.models.model_factory import create_model
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.eval()


def load_external_data(csv_path: Path, class_to_idx: dict):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must have image_path and label")
    paths = []
    labels = []
    for _, row in df.iterrows():
        p = ROOT / row["image_path"] if not Path(row["image_path"]).is_absolute() else Path(row["image_path"])
        if not p.exists():
            continue
        lbl = str(row["label"]).strip()
        if lbl not in class_to_idx:
            continue
        paths.append(p)
        labels.append(class_to_idx[lbl])
    return paths, np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-metrics", required=True, help="Path to internal test metrics.json")
    parser.add_argument("--external-csv", required=True, help="CSV with image_path, label (held-out domain)")
    parser.add_argument("--weights", required=True, help="Path to best_weights.pt")
    parser.add_argument("--model-name", default="EfficientNetB0")
    parser.add_argument("--out", default="ml/experiments/external_validation_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    root = ROOT
    internal_path = root / args.internal_metrics
    external_csv = root / args.external_csv
    weights_path = root / args.weights
    out_path = root / args.out

    if not internal_path.exists():
        raise FileNotFoundError(internal_path)
    if not external_csv.exists():
        raise FileNotFoundError(external_csv)
    if not weights_path.exists():
        raise FileNotFoundError(weights_path)

    class_names = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)

    internal = json.loads(internal_path.read_text(encoding="utf-8"))
    paths, y_true = load_external_data(external_csv, class_to_idx)
    if len(paths) == 0:
        raise ValueError("No valid rows in external CSV")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    model = load_model(weights_path, args.model_name, num_classes).to(args.device)

    y_prob = []
    with torch.no_grad():
        for i in range(0, len(paths), args.batch_size):
            batch_paths = paths[i : i + args.batch_size]
            batch = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                batch.append(transform(img))
            x = torch.stack(batch).to(args.device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.append(probs)
    y_prob = np.concatenate(y_prob, axis=0)
    y_pred = y_prob.argmax(axis=1)

    acc_ext = float(accuracy_score(y_true, y_pred))
    f1_ext = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_ext = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        roc_ext = 0.0

    acc_int = internal.get("accuracy", 0)
    f1_int = internal.get("f1_weighted", 0)
    roc_int = internal.get("roc_auc_ovr", 0)

    report = {
        "internal": {"accuracy": acc_int, "f1_weighted": f1_int, "roc_auc_ovr": roc_int},
        "external": {"accuracy": acc_ext, "f1_weighted": f1_ext, "roc_auc_ovr": roc_ext},
        "gap": {
            "accuracy_drop": round(acc_int - acc_ext, 4),
            "f1_drop": round(f1_int - f1_ext, 4),
            "roc_drop": round(roc_int - roc_ext, 4),
        },
        "external_n": int(len(paths)),
        "model": args.model_name,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report saved to", out_path)
    print("Internal accuracy:", acc_int, "| External accuracy:", acc_ext, "| Drop:", report["gap"]["accuracy_drop"])
