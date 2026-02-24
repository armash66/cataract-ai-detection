"""Run multi-class ONNX inference (for fundus_multi_disease.onnx when available)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["Cataract", "Glaucoma", "Diabetic Retinopathy", "Normal"]

_session = None
_session_path = None


def _preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...].astype(np.float32)


def run_onnx(onnx_path: str, image_path: str):
    """Run ONNX model at onnx_path on image; return (label, confidence_0_to_100)."""
    global _session, _session_path
    try:
        import onnxruntime as ort
    except ImportError:
        return None, None
    path = Path(onnx_path)
    if not path.exists():
        return None, None
    if _session_path != str(path):
        _session_path = str(path)
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = _preprocess(image_path)
    out = _session.run(None, {_session.get_inputs()[0].name: x})
    logits = out[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx] * 100)
