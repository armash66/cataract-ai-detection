"""Evaluation utilities for multi-class ophthalmology models."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob=None):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": p,
        "recall_weighted": r,
        "f1_weighted": f1,
    }
    if y_prob is not None:
        out["roc_auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    return out
