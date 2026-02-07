# CataractGPT: Research-Grade Cataract Screening and Explainability Suite

**Status:** Research prototype  
**Scope:** Educational and research use only (not a medical diagnostic device)

---

## Abstract
CataractGPT is a research-oriented system for cataract screening from ocular imagery. It integrates a Flask-based inference UI, fundus/anterior pipelines, and Grad-CAM explainability. The project emphasizes transparency, reproducibility, and experiment-driven iteration, and is intended for academic exploration rather than clinical deployment.

---

## Contributions
- End-to-end workflow: acquisition → inference → explainability → review
- Separate pipelines for fundus and anterior modalities
- Grad-CAM overlay with opacity and visualization controls
- Lightweight history persistence with SQLite
- Research-style interface for inspection and reporting

---

## Repository Structure
```
Cataract Detection/
├─ eye_web/              # Flask web app (inference UI)
│  ├─ app.py             # Server entrypoint
│  ├─ templates/         # HTML templates
│  ├─ static/            # CSS/JS/assets/uploads
│  ├─ inference.py       # Inference helper
│  └─ history.db         # SQLite history
├─ fundus_pipeline/      # Fundus training/eval utilities
├─ anterior_pipeline/    # Anterior pipeline utilities
├─ uploads/              # Stored images
├─ ui_v2/                # UI prototypes
├─ eyegpt_web/           # React UI prototype
└─ requirements.txt
```

---

## System Overview
**Data Flow**
1. Image acquisition (upload or camera)
2. Preprocessing and model inference
3. Prediction + confidence estimation
4. Grad-CAM explainability overlay
5. Persistence to local history

**Primary Outputs**
- Class prediction (Normal / Cataract)
- Confidence score
- Grad-CAM attention map

---

## Quick Start
```bash
pip install -r requirements.txt
cd eye_web
python app.py
```
Open: `http://127.0.0.1:5000`

---

## Pipelines
### Fundus Pipeline
`fundus_pipeline/` contains training, evaluation, and dataset tooling for fundus imagery.

### Anterior Pipeline
`anterior_pipeline/` contains preprocessing and model utilities tailored to anterior segment imagery.

---

## UI Components
- **Main Analysis:** upload, prediction, Grad-CAM overlay
- **History:** record review with details and exports
- **Protocols/Settings:** research dashboards (UI-only)

---

## Storage
- Results persisted to `eye_web/history.db`
- Images stored under `eye_web/static/uploads/`

---

## Limitations
- Not clinically validated
- Performance depends on capture conditions
- Explainability is indicative, not diagnostic

---

## Roadmap
- Modality-aware routing (fundus vs anterior)
- Automated quality gating (blur/glare)
- Report export (PDF)
- Model benchmarking + calibration reports

---

## Disclaimer
CataractGPT is a **research prototype** and **not a medical device**. Use only for educational and experimental purposes.

---
