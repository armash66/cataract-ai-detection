"""Multi-model training and benchmarking for retinal disease classification.

Expected directory layout:
ml/experiments/phase1/cleaned/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
  test/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.training.losses import make_loss
from ml.models.model_factory import (
    MODEL_NAMES,
    count_trainable_params,
    create_model,
    freeze_backbone,
    unfreeze_all,
)

CANONICAL_CLASSES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="ml/experiments/phase1/cleaned")
    parser.add_argument("--output-dir", default="ml/experiments/phase2")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-freeze", type=int, default=2)
    parser.add_argument("--epochs-finetune", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--loss", default="CrossEntropy", choices=["CrossEntropy", "FocalLoss"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]
    )
    eval_tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return train_tfms, eval_tfms


def get_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    train_tfms, eval_tfms = get_transforms()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes


def compute_class_weights(loader: DataLoader, num_classes: int, device: str) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, y in loader:
        y_np = y.numpy()
        for idx in y_np:
            counts[idx] += 1
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device: str, train_mode: bool):
    model.train(train_mode)
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train_mode:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train_mode:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total += y.size(0)
        correct += (preds == y).sum().item()

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def collect_predictions(model, loader, device: str):
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def save_training_curve(history: Dict[str, List[float]], out_file: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], out_file: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(classes)), labels=classes, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(classes)), labels=classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def save_roc(y_true: np.ndarray, y_prob: np.ndarray, classes: List[str], out_file: Path) -> float:
    y_true_bin = label_binarize(y_true, classes=list(range(len(classes))))
    auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")

    plt.figure(figsize=(6, 5))
    for idx, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        plt.plot(fpr, tpr, label=f"{cls}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()

    return float(auc)


def train_single_model(
    model_name: str,
    loaders,
    output_dir: Path,
    device: str,
    epochs_freeze: int,
    epochs_finetune: int,
    lr: float,
    patience: int,
    loss_name: str,
):
    train_loader, val_loader, test_loader, classes = loaders
    num_classes = len(classes)

    model = create_model(model_name, num_classes=num_classes, pretrained=True).to(device)

    class_weights = compute_class_weights(train_loader, num_classes, device)
    criterion = make_loss(loss_name, class_weights)

    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)
    best_weights = model_out / "best_weights.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")
    wait = 0

    # Stage 1: frozen backbone
    freeze_backbone(model, model_name)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    total_epochs = epochs_freeze + epochs_finetune

    for epoch in range(total_epochs):
        if epoch == epochs_freeze:
            unfreeze_all(model)
            optimizer = AdamW(model.parameters(), lr=lr * 0.3)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), best_weights)
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(torch.load(best_weights, map_location=device))
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted"))
    roc_auc = save_roc(y_true, y_prob, classes, model_out / "roc_curve.png")

    save_confusion(y_true, y_pred, classes, model_out / "confusion_matrix.png")
    save_training_curve(history, model_out / "training_curve.png")

    model_size_mb = best_weights.stat().st_size / (1024 * 1024)
    params = count_trainable_params(model)

    metrics = {
        "model": model_name,
        "classes": classes,
        "accuracy": acc,
        "f1_weighted": f1,
        "roc_auc_ovr": roc_auc,
        "params_trainable": int(params),
        "model_size_mb": round(float(model_size_mb), 4),
        "loss": loss_name,
        "epochs_run": len(history["train_loss"]),
        "best_weights": str(best_weights),
    }

    (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def save_comparison(rows: List[Dict], out_csv: Path) -> None:
    fields = ["Model", "Accuracy", "F1", "ROC-AUC", "Params", "Model Size"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["model"],
                    "Accuracy": f"{row['accuracy']:.4f}",
                    "F1": f"{row['f1_weighted']:.4f}",
                    "ROC-AUC": f"{row['roc_auc_ovr']:.4f}",
                    "Params": row["params_trainable"],
                    "Model Size": f"{row['model_size_mb']:.4f} MB",
                }
            )


def main() -> None:
    args = parse_args()
    device = args.device

    root = Path(__file__).resolve().parents[2]
    data_dir = root / args.data_dir
    output_dir = root / args.output_dir

    if not (data_dir / "train").exists() or not (data_dir / "val").exists() or not (data_dir / "test").exists():
        raise FileNotFoundError(f"Expected train/val/test folders under {data_dir}")

    loaders = get_dataloaders(data_dir, args.batch_size, args.num_workers)
    classes = loaders[3]
    if set(classes) != set(c.replace(" ", "_") for c in CANONICAL_CLASSES):
        print("Warning: class names differ from canonical mapping.")

    results = []
    for model_name in args.models:
        print(f"Training {model_name} on {device}...")
        metrics = train_single_model(
            model_name=model_name,
            loaders=loaders,
            output_dir=output_dir,
            device=device,
            epochs_freeze=args.epochs_freeze,
            epochs_finetune=args.epochs_finetune,
            lr=args.lr,
            patience=args.patience,
            loss_name=args.loss,
        )
        results.append(metrics)

    save_comparison(results, output_dir / "model_comparison.csv")
    (output_dir / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved comparison:", output_dir / "model_comparison.csv")


if __name__ == "__main__":
    main()
