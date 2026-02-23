"""Clean image set with blur and brightness quality gates.

Reads a split CSV from merge_and_normalize.py and writes cleaned images to:
- ml/experiments/phase1/cleaned/<split>/<class>/<filename>

Also writes a quality report:
- ml/experiments/phase1/quality_report_<split>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd

TARGET_SIZE = (224, 224)
LAPLACIAN_VAR_THRESHOLD = 60.0
LOW_BRIGHTNESS_THRESHOLD = 35.0
HIGH_BRIGHTNESS_THRESHOLD = 220.0


def variance_of_laplacian(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def quality_flags(img_bgr: np.ndarray) -> Dict[str, bool]:
    lap_var = variance_of_laplacian(img_bgr)
    bright = mean_brightness(img_bgr)
    return {
        "is_blurry": lap_var < LAPLACIAN_VAR_THRESHOLD,
        "too_dark": bright < LOW_BRIGHTNESS_THRESHOLD,
        "too_bright": bright > HIGH_BRIGHTNESS_THRESHOLD,
        "laplacian_var": lap_var,
        "brightness_mean": bright,
    }


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True, help="CSV from phase1 splits (train/val/test)")
    parser.add_argument("--out-dir", default="ml/experiments/phase1/cleaned")
    parser.add_argument("--copy-invalid", action="store_true", help="If set, keep invalid images in invalid/ for inspection")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    split_csv = root / args.split_csv
    out_dir = root / args.out_dir

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    required = {"image_path", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {split_csv}")

    report = {
        "input_csv": str(split_csv),
        "target_size": list(TARGET_SIZE),
        "thresholds": {
            "laplacian_var_min": LAPLACIAN_VAR_THRESHOLD,
            "brightness_min": LOW_BRIGHTNESS_THRESHOLD,
            "brightness_max": HIGH_BRIGHTNESS_THRESHOLD,
        },
        "processed": 0,
        "saved": 0,
        "rejected": 0,
        "reject_reasons": {"missing_file": 0, "blurry": 0, "too_dark": 0, "too_bright": 0},
    }

    split_name = split_csv.stem

    for row in df.itertuples(index=False):
        report["processed"] += 1
        src = Path(str(row.image_path))
        if not src.is_absolute():
            src = (root / src).resolve()

        if not src.exists():
            report["rejected"] += 1
            report["reject_reasons"]["missing_file"] += 1
            continue

        img = cv2.imread(str(src))
        if img is None:
            report["rejected"] += 1
            report["reject_reasons"]["missing_file"] += 1
            continue

        flags = quality_flags(img)
        invalid = flags["is_blurry"] or flags["too_dark"] or flags["too_bright"]

        safe_label = str(row.label).replace(" ", "_")
        rel_name = src.name

        if invalid:
            report["rejected"] += 1
            if flags["is_blurry"]:
                report["reject_reasons"]["blurry"] += 1
            if flags["too_dark"]:
                report["reject_reasons"]["too_dark"] += 1
            if flags["too_bright"]:
                report["reject_reasons"]["too_bright"] += 1

            if args.copy_invalid:
                invalid_dir = out_dir / "invalid" / split_name / safe_label
                invalid_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(invalid_dir / rel_name), img)
            continue

        clean = preprocess(img)
        target_dir = out_dir / split_name / safe_label
        target_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target_dir / rel_name), clean)
        report["saved"] += 1

    report_file = out_dir.parent / f"quality_report_{split_name}.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved quality report:", report_file)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
