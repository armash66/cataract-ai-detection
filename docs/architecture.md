# EyeGPT-AI Architecture

## Identity
EyeGPT-AI is a research-grade multi-architecture retinal disease benchmark with explainability and browser-ready deployment.

## Pipeline
1. Data merge, normalization, and quality gating (`ml/data/`).
2. Multi-model training and benchmarking (`ml/models/`, `ml/training/`, `ml/evaluation/`).
3. Research studies (cross-validation and ablation) in `ml/experiments/`.
4. Explainability generation via Grad-CAM (`ml/explainability/`).
5. Export and deployment packaging (`ml/export/`, `model_registry/`).
6. Browser inference and interactive analysis UI (`frontend/`).

## Core Models
- Transfer learning: EfficientNetB0, ResNet50, ViT, EfficientNetV2, ConvNeXt, MobileNetV3
- Custom lightweight: EyeGPTNet (<5M parameters target)

## Deployment Contract
- Input: RGB image, 224x224 normalized to [0,1], NCHW
- Output: logits for 4 classes (Cataract, Glaucoma, Diabetic Retinopathy, Normal)
- Browser runtime: `onnxruntime-web`
