# Dataset Plan

## Sources
- Kaggle: `h100gpu/eye-disease-detection`
- Kaggle: `gayatrinadar/eye-disease-detection`

## Unified Labels
- Cataract
- Glaucoma
- Diabetic Retinopathy
- Normal

## Phase 1 Pipeline

1. Merge and normalize labels:
```bash
python ml/data/merge_and_normalize.py \
  --csv ml/experiments/dataset1_labels.csv ml/experiments/dataset2_labels.csv \
  --out-dir ml/experiments/phase1
```

2. Clean each split with quality gates (blur + brightness):
```bash
python ml/data/clean_images.py --split-csv ml/experiments/phase1/splits/train.csv
python ml/data/clean_images.py --split-csv ml/experiments/phase1/splits/val.csv
python ml/data/clean_images.py --split-csv ml/experiments/phase1/splits/test.csv
```

3. Use augmentation module in training loaders:
- `ml/data/augmentations.py`

## Split Strategy
- Train: 70%
- Validation: 15%
- Test: 15%
- Split is stratified by class.

## Outputs
- `ml/experiments/phase1/merged_labels.csv`
- `ml/experiments/phase1/splits/train.csv`
- `ml/experiments/phase1/splits/val.csv`
- `ml/experiments/phase1/splits/test.csv`
- `ml/experiments/phase1/class_distribution.png`
- `ml/experiments/phase1/dataset_summary.json`
- `ml/experiments/phase1/quality_report_train.json`
- `ml/experiments/phase1/quality_report_val.json`
- `ml/experiments/phase1/quality_report_test.json`
