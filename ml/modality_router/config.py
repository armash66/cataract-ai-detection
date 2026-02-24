"""
Modality router configuration.

Defines which modalities exist and which model path to use per modality.
Update these paths when you export fundus_multi_disease and anterior_multi_disease ONNX models.
"""

from pathlib import Path

# Supported imaging modalities
MODALITIES = ("fundus", "anterior")

# Default modality when user does not select (e.g. backend default)
DEFAULT_MODALITY = "anterior"

# Model paths relative to project root (for server-side / training).
# For frontend, use /models/<name>.onnx; these are used by eye_web and export scripts.
MODALITY_MODEL_PATHS = {
    "fundus": "model_registry/fundus_multi_disease.onnx",
    "anterior": "model_registry/anterior_multi_disease.onnx",
}

# Fallback when modality-specific ONNX not yet exported (single consolidated model)
FALLBACK_MODEL_PATH = "model_registry/best_accuracy.onnx"


def get_model_path_for_modality(modality: str, project_root: Path) -> Path:
    """Return the ONNX or .pt path for the given modality. Prefers modality-specific, else fallback."""
    modality = (modality or "").strip().lower()
    if modality not in MODALITIES:
        modality = DEFAULT_MODALITY
    path = MODALITY_MODEL_PATHS.get(modality) or FALLBACK_MODEL_PATH
    return project_root / path
