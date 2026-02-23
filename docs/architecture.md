# EyeGPT-AI Architecture

## System Diagram

```mermaid
flowchart TD
  A[Dataset Sources] --> B[ml/data]
  B --> C[ml/models + ml/training]
  C --> D[ml/evaluation]
  C --> E[ml/explainability]
  C --> F[ml/export]
  D --> G[model_registry]
  E --> G
  F --> G
  G --> H[frontend inference + UI]
```

## Components
- `ml/data`: merge, normalize, quality filtering, split and summaries
- `ml/models`: transfer-learning backbones + custom EyeGPTNet
- `ml/training`: benchmark, cross-validation, ablation
- `ml/evaluation`: metrics, confusion, ROC, benchmark helpers
- `ml/explainability`: Grad-CAM and heatmap export tools
- `ml/export`: ONNX/quantized export and artifact benchmarking
- `model_registry`: stable deployment artifacts and metadata
- `frontend`: EyeGPT browser inference interface
