"""Loss functions for class imbalance handling."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        target_log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - target_prob).pow(self.gamma)
        loss = -focal_term * target_log_prob

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = loss * alpha_t

        return loss.mean()


def make_loss(name: str, class_weights: torch.Tensor | None = None) -> nn.Module:
    if name.lower() == "focalloss":
        return FocalLoss(alpha=class_weights)
    if name.lower() == "crossentropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    raise ValueError(f"Unknown loss: {name}")
