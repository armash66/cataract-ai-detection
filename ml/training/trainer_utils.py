"""Reusable utilities for training/evaluation loops."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
