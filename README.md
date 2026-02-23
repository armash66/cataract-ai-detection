# EyeGPT-AI

Research-grade multi-architecture retinal disease classification benchmark with explainable AI and browser-ready deployment.

## Scope
- Multi-disease classes: Cataract, Glaucoma, Diabetic Retinopathy, Normal
- Multi-model training and benchmarking
- Custom lightweight model design: EyeGPTNet
- Explainability (Grad-CAM)
- ONNX export and browser inference path

## Repository Structure
```text
EyeGPT-AI/
+-- ml/
¦   +-- data/
¦   +-- models/
¦   +-- training/
¦   +-- evaluation/
¦   +-- explainability/
¦   +-- export/
¦   +-- experiments/
¦   +-- utils/
+-- model_registry/
+-- frontend/
+-- docs/
+-- README.md
```

## Training Workflow
1. Build dataset manifests and splits:
```bash
python ml/data/merge_and_normalize.py
python ml/data/clean_images.py --split-csv ml/experiments/phase1/splits/train.csv
python ml/data/clean_images.py --split-csv ml/experiments/phase1/splits/val.csv
python ml/data/clean_images.py --split-csv ml/experiments/phase1/splits/test.csv
```
2. Train and benchmark models:
```bash
python ml/training/train_and_benchmark.py --models EfficientNetB0 ResNet50 ViT EyeGPTNet
```
3. Run research studies:
```bash
python ml/training/cross_validation.py
python ml/training/ablation_study.py
```

## Export Workflow
```bash
python ml/export/export_models.py
python ml/evaluation/performance_benchmark.py
```
Artifacts are written to `model_registry/`.

## Frontend (EyeGPT)
```bash
cd frontend
npm install
npm run dev
```
- Uses real ONNX browser inference when `/models/best_model.onnx` is available.
- Falls back to deterministic mock inference if model/runtime is unavailable.

## Explainability
- Grad-CAM generation: `ml/explainability/gradcam.py`
- Transparent heatmaps: `ml/explainability/heatmap_utils.py`
- UI overlay controls: `frontend/src/components/ImagingPanel.jsx`

## Disclaimer
Educational and research project only. Not a medical device and not a clinical diagnostic system.
