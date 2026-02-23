from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ml.explainability.gradcam import GradCAM
from ml.explainability.heatmap_utils import save_transparent_heatmap


def _pick_target_layer(model: torch.nn.Module):
    # Generic fallback: last Conv2d layer
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM target layer selection.")
    return last_conv


def generate_example_heatmaps(model: torch.nn.Module, image_paths: list[Path], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    cam_gen = GradCAM(model, _pick_target_layer(model))

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        cam = cam_gen(x)
        save_transparent_heatmap(cam, out_dir / f"{p.stem}_gradcam.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=False, default=[])
    parser.add_argument("--out-dir", default="docs/explainability")
    args = parser.parse_args()

    # Lightweight default model for offline generation path
    from ml.models.model_factory import create_model

    model = create_model("EyeGPTNet", num_classes=4, pretrained=False)
    images = [Path(i) for i in args.images]
    out_dir = Path(args.out_dir)

    if not images:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Create transparent placeholder if no images provided
        blank = np.zeros((224, 224), dtype=np.float32)
        save_transparent_heatmap(blank, out_dir / "placeholder_gradcam.png")
        print("Generated placeholder heatmap in", out_dir)
        return

    generate_example_heatmaps(model, images, out_dir)
    print("Saved explainability outputs to", out_dir)


if __name__ == "__main__":
    main()
