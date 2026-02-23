# Model Comparison Notes

Model comparison artifacts are generated under:
- `ml/experiments/phase2/model_comparison.csv`
- `model_registry/model_comparison.csv` (stable deployment copy)

Tracked metrics:
- Accuracy
- F1
- ROC-AUC
- Parameter count
- Model size
- Inference latency (benchmark scripts)

Use these metrics to choose:
- Best-accuracy model for research demo
- Smallest low-latency model for browser deployment
