"""Merge and normalize multi-source eye disease datasets.

Expected input CSV columns:
- image_path: path to image file (absolute or relative)
- label: raw label string
- patient_id (optional): same patient must not appear in more than one of train/val/test

When patient_id is provided, splits are done at patient level (no leakage).

Outputs:
- ml/experiments/phase1/merged_labels.csv
- ml/experiments/phase1/splits/{train,val,test}.csv
- ml/experiments/phase1/class_distribution.png
- ml/experiments/phase1/dataset_summary.json
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupShuffleSplit

CANONICAL_CLASSES = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]

CLASS_MAP = {
    "cataract": "Cataract",
    "glaucoma": "Glaucoma",
    "diabetic retinopathy": "Diabetic Retinopathy",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "dr": "Diabetic Retinopathy",
    "normal": "Normal",
}


def normalize_label(raw: str) -> str | None:
    key = str(raw).strip().lower()
    return CLASS_MAP.get(key)


def stable_id(path_value: str) -> str:
    return hashlib.sha1(path_value.encode("utf-8")).hexdigest()


def load_and_normalize(csv_paths: List[Path], patient_id_column: Optional[str] = None) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        required = {"image_path", "label"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {csv_path}")

        use_cols = ["image_path", "label"]
        if patient_id_column and patient_id_column in df.columns:
            use_cols.append(patient_id_column)
        df = df[use_cols].copy()
        df["label"] = df["label"].apply(normalize_label)
        df["source_csv"] = str(csv_path)
        df = df[df["label"].isin(CANONICAL_CLASSES)]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No valid CSV inputs found. Provide --csv paths.")

    merged = pd.concat(frames, ignore_index=True)
    merged["image_path"] = merged["image_path"].astype(str)
    merged["sample_id"] = merged["image_path"].apply(stable_id)

    if patient_id_column and patient_id_column in merged.columns:
        merged["patient_id"] = merged[patient_id_column].astype(str)
    else:
        # No patient-level info: treat each image as its own "patient" to keep API consistent
        merged["patient_id"] = merged["sample_id"]

    # Duplicate removal by normalized path key.
    before = len(merged)
    merged = merged.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    after = len(merged)
    merged.attrs["duplicates_removed"] = int(before - after)
    merged.attrs["split_by_patient"] = bool(patient_id_column and patient_id_column in merged.columns)
    return merged


def stratified_split(df: pd.DataFrame, by_patient: bool = False) -> Dict[str, pd.DataFrame]:
    """Split into train/val/test. If by_patient, use StratifiedGroupShuffleSplit so no patient leaks across splits."""
    if not by_patient or "patient_id" not in df.columns:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=df["label"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            stratify=temp_df["label"],
        )
        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

    # Patient-level split: same patient never in two splits
    groups = df["patient_id"].values
    y = df["label"].values
    sgss = StratifiedGroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sgss.split(df, y, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Stratified split of temp into val/test by patient
    temp_groups = temp_df["patient_id"].values
    temp_y = temp_df["label"].values
    sgss2 = StratifiedGroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(sgss2.split(temp_df, temp_y, groups=temp_groups))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def make_distribution_chart(df: pd.DataFrame, out_file: Path) -> Dict[str, int]:
    counts = df["label"].value_counts().reindex(CANONICAL_CLASSES, fill_value=0)
    plt.figure(figsize=(8, 4.5))
    counts.plot(kind="bar", color=["#1d8f6d", "#2a6fba", "#ef8a17", "#656d78"])
    plt.title("Class Distribution (Merged)")
    plt.ylabel("Samples")
    plt.xlabel("Class")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=160)
    plt.close()
    return {k: int(v) for k, v in counts.to_dict().items()}


def imbalance_report(class_counts: Dict[str, int]) -> Dict[str, float]:
    nonzero = [v for v in class_counts.values() if v > 0]
    ratio = max(nonzero) / min(nonzero) if nonzero else 0.0
    return {
        "max_to_min_ratio": round(float(ratio), 4),
        "is_imbalanced_ratio_gt_1_5": bool(ratio > 1.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "ml/experiments/dataset1_labels.csv",
            "ml/experiments/dataset2_labels.csv",
        ],
        help="Input CSV files with columns image_path,label",
    )
    parser.add_argument(
        "--out-dir",
        default="ml/experiments/phase1",
        help="Output directory for merged files and reports",
    )
    parser.add_argument(
        "--patient-id-column",
        default=None,
        help="CSV column name for patient/subject ID. When set, train/val/test splits are at patient level (no leakage).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    csv_paths = [root / p for p in args.csv]
    out_dir = root / args.out_dir
    split_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    merged = load_and_normalize(csv_paths, patient_id_column=args.patient_id_column)
    merged_file = out_dir / "merged_labels.csv"
    merged.to_csv(merged_file, index=False)

    by_patient = bool(merged.attrs.get("split_by_patient", False))
    splits = stratified_split(merged, by_patient=by_patient)
    split_stats = {}
    for split_name, split_df in splits.items():
        split_path = split_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        split_stats[split_name] = {
            "samples": int(len(split_df)),
            "class_counts": {k: int(v) for k, v in split_df["label"].value_counts().to_dict().items()},
        }

    chart_file = out_dir / "class_distribution.png"
    class_counts = make_distribution_chart(merged, chart_file)

    summary = {
        "total_samples": int(len(merged)),
        "classes": CANONICAL_CLASSES,
        "duplicates_removed": int(merged.attrs.get("duplicates_removed", 0)),
        "split_by_patient": bool(merged.attrs.get("split_by_patient", False)),
        "class_counts": class_counts,
        "imbalance": imbalance_report(class_counts),
        "splits": split_stats,
        "outputs": {
            "merged_csv": str(merged_file),
            "distribution_chart": str(chart_file),
            "split_dir": str(split_dir),
        },
    }

    summary_file = out_dir / "dataset_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved merged dataset to:", merged_file)
    print("Saved summary to:", summary_file)
    print("Saved class chart to:", chart_file)


if __name__ == "__main__":
    main()
