"""Augmentation utilities for retinal training.

Provides a composable Albumentations pipeline with:
- rotation
- brightness/contrast
- zoom
- CLAHE
- gaussian noise

MixUp is provided as a helper for training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install albumentations to use this module: pip install albumentations") from exc


@dataclass(frozen=True)
class AugmentationConfig:
    rotation_limit: int = 20
    brightness_limit: float = 0.25
    contrast_limit: float = 0.25
    zoom_scale_min: float = 0.9
    zoom_scale_max: float = 1.1
    gaussian_noise_std: float = 0.03
    clahe_clip_limit: float = 2.0
    mixup_alpha: float = 0.2


def build_train_augmentations(cfg: AugmentationConfig | None = None) -> A.Compose:
    cfg = cfg or AugmentationConfig()
    return A.Compose(
        [
            A.Rotate(limit=cfg.rotation_limit, p=0.6),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.6,
            ),
            A.RandomResizedCrop(
                height=224,
                width=224,
                scale=(cfg.zoom_scale_min, cfg.zoom_scale_max),
                ratio=(0.95, 1.05),
                p=0.5,
            ),
            A.CLAHE(clip_limit=cfg.clahe_clip_limit, p=0.45),
            A.GaussNoise(std_range=(0.0, cfg.gaussian_noise_std), p=0.35),
        ]
    )


def build_eval_augmentations() -> A.Compose:
    return A.Compose([])


def apply_mixup(
    images_a: np.ndarray,
    labels_a: np.ndarray,
    images_b: np.ndarray,
    labels_b: np.ndarray,
    alpha: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    if alpha <= 0:
        return images_a, labels_a

    lam = np.random.beta(alpha, alpha)
    mixed_images = lam * images_a + (1.0 - lam) * images_b
    mixed_labels = lam * labels_a + (1.0 - lam) * labels_b
    return mixed_images.astype(images_a.dtype), mixed_labels.astype(labels_a.dtype)
