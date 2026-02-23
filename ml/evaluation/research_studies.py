"""Phase 3 research studies: cross-validation + ablation experiments.

Outputs:
- ml/experiments/phase3/crossval_results.csv
- ml/experiments/phase3/ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import io, transforms

from ml.training.losses import make_loss
from ml.training.model_factory import create_model, freeze_backbone, unfreeze_all

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label: int


class PathDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = io.read_image(str(sample.path)).float() / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return img, sample.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ml/experiments/phase1/cleaned")
    parser.add_argument("--out-dir", default="ml/experiments/phase3")
    parser.add_argument("--model", default="EfficientNetB0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def collect_samples(cleaned_dir: Path) -> Tuple[List[Sample], List[str]]:
    samples: List[Sample] = []

    class_names = set()
    for split in ["train", "val", "test"]:
        split_dir = cleaned_dir / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_names.add(class_dir.name)

    if not class_names:
        raise FileNotFoundError(f"No class folders found under {cleaned_dir}")

    classes = sorted(class_names)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for split in ["train", "val", "test"]:
        split_dir = cleaned_dir / split
        if not split_dir.exists():
            continue
        for cls in classes:
            class_dir = split_dir / cls
            if not class_dir.exists():
                continue
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    samples.append(Sample(path=p, label=class_to_idx[cls]))

    if not samples:
        raise FileNotFoundError("No images found in cleaned dataset structure.")

    return samples, classes


def make_transforms(use_augmentation: bool):
    base = [transforms.Resize((224, 224))]
    aug = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    return transforms.Compose(base + aug if use_augmentation else base)


def make_loader(
    ds: Dataset,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    weighted_sampling: bool,
):
    if not weighted_sampling:
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    class_counts = np.bincount(labels)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)


def evaluate(model, loader, device: str):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def run_train_eval(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    num_classes: int,
    model_name: str,
    freeze_mode: bool,
    use_augmentation: bool,
    weighted_sampling: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
):
    train_labels = np.array([s.label for s in train_samples])

    train_ds = PathDataset(train_samples, transform=make_transforms(use_augmentation))
    val_ds = PathDataset(val_samples, transform=make_transforms(False))

    train_loader = make_loader(train_ds, train_labels, batch_size, num_workers, weighted_sampling)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = create_model(model_name, num_classes=num_classes, pretrained=True).to(device)

    if freeze_mode:
        freeze_backbone(model, model_name)
    else:
        unfreeze_all(model)

    class_weights = None
    if weighted_sampling:
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        class_weights = torch.tensor(counts.sum() / (num_classes * counts), dtype=torch.float32, device=device)

    criterion = make_loss("CrossEntropy", class_weights)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    return evaluate(model, val_loader, device)


def run_crossval(
    samples: Sequence[Sample],
    model_name: str,
    k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
):
    labels = np.array([s.label for s in samples])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        metrics = run_train_eval(
            train_samples,
            val_samples,
            num_classes=len(set(labels.tolist())),
            model_name=model_name,
            freeze_mode=True,
            use_augmentation=True,
            weighted_sampling=True,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_workers=num_workers,
        )
        rows.append({"fold": fold, "model": model_name, **metrics})

    return rows


def run_ablation(
    samples: Sequence[Sample],
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    num_workers: int,
):
    labels = np.array([s.label for s in samples])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    train_idx, val_idx = next(iter(skf.split(np.zeros(len(labels)), labels)))

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    configs = [
        {"name": "Frozen_WithAug_Weighted", "freeze": True, "aug": True, "weighted": True},
        {"name": "Finetune_WithAug_Weighted", "freeze": False, "aug": True, "weighted": True},
        {"name": "Frozen_NoAug_Weighted", "freeze": True, "aug": False, "weighted": True},
        {"name": "Frozen_WithAug_Unweighted", "freeze": True, "aug": True, "weighted": False},
    ]

    rows = []
    for cfg in configs:
        metrics = run_train_eval(
            train_samples,
            val_samples,
            num_classes=len(set(labels.tolist())),
            model_name=model_name,
            freeze_mode=cfg["freeze"],
            use_augmentation=cfg["aug"],
            weighted_sampling=cfg["weighted"],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_workers=num_workers,
        )
        rows.append(
            {
                "experiment": cfg["name"],
                "model": model_name,
                "freeze_backbone": cfg["freeze"],
                "use_augmentation": cfg["aug"],
                "weighted_sampling": cfg["weighted"],
                **metrics,
            }
        )

    return rows


def save_csv(rows: Sequence[dict], out_file: Path):
    if not rows:
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, classes = collect_samples(data_dir)
    print(f"Loaded {len(samples)} images across {len(classes)} classes")

    crossval_rows = run_crossval(
        samples=samples,
        model_name=args.model,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
    )
    save_csv(crossval_rows, out_dir / "crossval_results.csv")

    ablation_rows = run_ablation(
        samples=samples,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        num_workers=args.num_workers,
    )
    save_csv(ablation_rows, out_dir / "ablation_results.csv")

    print("Saved:", out_dir / "crossval_results.csv")
    print("Saved:", out_dir / "ablation_results.csv")


if __name__ == "__main__":
    main()
