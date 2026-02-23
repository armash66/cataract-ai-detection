# Research Notes

## Completed
- Stratified split and quality filtering
- Multi-model baseline + advanced benchmarking
- Cross-validation and ablation studies

## Current Focus
- ONNX export and quantized export stability
- Grad-CAM export examples for each class
- Frontend inference calibration and confidence handling

## Reproducibility Checklist
- Seed all runs (`ml/utils/seed.py`)
- Save config per run (`ml/utils/config.py`)
- Persist metrics and artifacts under `ml/experiments/`
