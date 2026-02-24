"""Build metrics dashboard JSON from training artifacts (confusion matrix, ROC, per-class metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-dir", default="ml/experiments/phase2")
    parser.add_argument("--out", default="frontend/public/metrics/benchmark.json")
    args = parser.parse_args()

    phase2 = ROOT / args.phase2_dir
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = []
    if phase2.exists():
        for d in phase2.iterdir():
            if not d.is_dir():
                continue
            metrics_file = d / "metrics.json"
            if not metrics_file.exists():
                continue
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
            entry = {
                "model": data.get("model", d.name),
                "accuracy": data.get("accuracy", 0),
                "f1_weighted": data.get("f1_weighted", 0),
                "roc_auc_ovr": data.get("roc_auc_ovr", 0),
                "params_trainable": data.get("params_trainable"),
                "cpu_inference_ms": data.get("cpu_inference_ms"),
                "confusion_matrix_url": f"/metrics/{d.name}/confusion_matrix.png",
                "roc_curve_url": f"/metrics/{d.name}/roc_curve.png",
            }
            models.append(entry)

    payload = {"models": models, "source": str(phase2)}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved", out_path)


if __name__ == "__main__":
    main()
