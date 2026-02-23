from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ml.explainability.heatmap_utils import save_transparent_heatmap


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self.activations = output.detach()

        def bwd_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].backward()

        grads = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        return cam


def generate_gradcam_png(model, target_layer, image_path: str, out_png: str):
    model.eval()
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    cam_gen = GradCAM(model, target_layer)
    cam = cam_gen(tensor)
    save_transparent_heatmap(cam, Path(out_png))
