"""
Modality router: route images to the correct expert model (fundus vs anterior).

- Use a UI selector or a lightweight modality classifier to choose model.
- If modality is uncertain, return abstention instead of forcing a prediction.
"""

from .config import MODALITIES, get_model_path_for_modality

__all__ = ["MODALITIES", "get_model_path_for_modality"]
