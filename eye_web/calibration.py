"""Load calibration JSON for abstention threshold (and optional per-class)."""

from __future__ import annotations

import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CALIBRATION_PATHS = [
    PROJECT_ROOT / "model_registry" / "anterior_calibration.json",
    PROJECT_ROOT / "model_registry" / "calibration.json",
]

_loaded = None


def get_abstain_threshold() -> float:
    """Return global abstain confidence threshold (0–100 scale). Default 50."""
    global _loaded
    if _loaded is not None:
        return _loaded
    for p in CALIBRATION_PATHS:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                t = data.get("global_abstain_threshold")
                if t is not None:
                    val = float(t)
                    _loaded = (val * 100.0) if val <= 1.0 else val
                    return _loaded
            except Exception:
                pass
    _loaded = 50.0
    return 50.0
