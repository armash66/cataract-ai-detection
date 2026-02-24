# Runbook

Quick reference for modality, abstention, calibration, data splits, and new scripts.

## Modality

- **Frontend**: Image type selector (Fundus / Anterior) routes to `/models/fundus_multi_disease.onnx` or `/models/anterior_multi_disease.onnx`; fallback `/models/best_accuracy.onnx`.
- **Config**: `frontend/src/config/modality.js`, `ml/modality_router/config.py`.
- **Eye_web**: Form field `scan_type`; when fundus ONNX exists at `model_registry/fundus_multi_disease.onnx`, it is used; else anterior pipeline or abstain (modality mismatch).
- **Train/export**: `python -m ml.training.train_modality --modality fundus --data-dir ...`; `python -m ml.export.export_modality_onnx --modality fundus --weights ...`.

## Abstention

- **When**: Low confidence (< threshold), low image quality, or modality mismatch.
- **Frontend**: `/calibration.json` provides per-class and global thresholds; `inference.js` uses them for abstention.
- **Eye_web**: `calibration.py` loads `model_registry/anterior_calibration.json` or `calibration.json` for threshold.

## Calibration

- **Script**: `python -m ml.training.calibrate_and_thresholds --weights <path> --model-name EfficientNetB0 --out model_registry/calibration.json`
- **Output**: `temperature`, `per_class_thresholds`, `global_abstain_threshold` in JSON.

## Patient-level splits

- **Script**: `python ml/data/merge_and_normalize.py --csv ... --patient-id-column patient_id`
- **Effect**: Train/val/test splits by patient (no leakage). Summary has `split_by_patient: true`.

## External validation

- **Script**: `python -m ml.evaluation.external_validation --internal-metrics ... --external-csv ... --weights ... --out ...`
- **Output**: `external_validation_report.json` with internal vs external accuracy/F1/ROC and gap.

## Reliability tests

- **Script**: `python -m ml.evaluation.reliability_tests`
- **Covers**: fundus/anterior routing, low-confidence abstain, low-quality abstain, modality-mismatch handling, calibration threshold.

## Metrics dashboard

- **Build**: `python -m ml.evaluation.build_metrics_dashboard --phase2-dir ml/experiments/phase2 --out frontend/public/metrics/benchmark.json`
- **UI**: Analysis page loads `/metrics/benchmark.json` and shows table + confusion/ROC images when available.
