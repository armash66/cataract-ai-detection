from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    denom = cam.max() + 1e-8
    cam = cam / denom
    return cam


def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam = cv2.resize(cam, (image_bgr.shape[1], image_bgr.shape[0]))
    cam_u8 = np.uint8(255 * normalize_heatmap(cam))
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)
    return out


def save_transparent_heatmap(cam: np.ndarray, out_png: Path) -> None:
    cam_u8 = np.uint8(255 * normalize_heatmap(cam))
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    alpha = cam_u8
    rgba = np.dstack([heat[:, :, 2], heat[:, :, 1], heat[:, :, 0], alpha])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), rgba)
