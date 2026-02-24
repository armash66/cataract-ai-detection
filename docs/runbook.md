# Runbook

Quick reference for modality, abstention, calibration, and data splits.

## Modality

- **Frontend**: Image type selector (Fundus / Anterior) routes to `/models/fundus_multi_disease.onnx` or `/models/anterior_multi_disease.onnx`; fallback `/models/best_accuracy.onnx`.
- **Config**: `frontend/src/config/modality.js`, `ml/modality_router/config.py`.
- **Eye_web**: Form field `scan_type`; `inference.run_inference(image_path, scan_type=...)`.

## Abstention

- **When**: Low confidence (< threshold) or low image quality; or modality mismatch (e.g. fundus requested but only anterior model).
- **Frontend**: `frontend/src/utils/inference.js` (ABSTAIN_*), `PredictionDashboard.jsx` (banner + Uncertain badge).
- **Eye_web**: `eye_web/inference.py` returns `abstain: true` and message; `app.py` and `index.html` show message and Uncertain.

## Calibration

- **Script**: `python -m ml.training.calibrate_and_thresholds --weights <path> --model-name EfficientNetB0 --out model_registry/calibration.json`
- **Output**: `temperature`, `per_class_thresholds`, `global_abstain_threshold` in JSON.

## Patient-level splits

- **Script**: `python ml/data/merge_and_normalize.py --csv ... --patient-id-column patient_id`
- **Effect**: Train/val/test splits by patient (no leakage). Summary has `split_by_patient: true`.
