import os
import sys
import shutil
from quality import check_image_quality

# -------------------------------------------------
# Path setup (REQUIRED)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# Imports
# -------------------------------------------------
from anterior_pipeline.src.predict import predict_image
from anterior_pipeline.src.gradcam import generate_gradcam

# -------------------------------------------------
# Static output paths
# -------------------------------------------------
STATIC_DIR = os.path.join(BASE_DIR, "static")
STATIC_GEN_DIR = os.path.join(STATIC_DIR, "generated")
os.makedirs(STATIC_GEN_DIR, exist_ok=True)

# Abstention: do not force definitive output when confidence is too low
ABSTAIN_CONFIDENCE_THRESHOLD = 0.5
MODALITIES = ("fundus", "anterior")
DEFAULT_MODALITY = "anterior"


import uuid

def run_inference(image_path, scan_type=None):
    """
    Full inference pipeline:
    - Quality check
    - Modality routing (fundus vs anterior)
    - Prediction
    - Abstention when confidence too low
    - Grad-CAM generation

    scan_type: "fundus" | "anterior" (default). Routes to the appropriate expert model.
    Returns:
        prediction, confidence, paths, or error/abstain payload.
    """
    scan_type = (scan_type or "").strip().lower() or DEFAULT_MODALITY
    if scan_type not in MODALITIES:
        scan_type = DEFAULT_MODALITY

    # -----------------------------
    # Quality gate
    # -----------------------------
    ok, message = check_image_quality(image_path)
    if not ok:
        return {"error": message}

    # -----------------------------
    # Modality routing: anterior only for now; fundus can be added when model exists
    # -----------------------------
    if scan_type == "fundus":
        # Placeholder: when fundus_multi_disease model is deployed, call it here.
        # For now we still use anterior model and flag that fundus was requested.
        label, confidence = predict_image(image_path)
        modality_mismatch = True  # User said fundus but we used anterior
    else:
        label, confidence = predict_image(image_path)
        modality_mismatch = False

    # -----------------------------
    # Abstention: never force definitive disease output on bad inputs
    # -----------------------------
    if confidence < ABSTAIN_CONFIDENCE_THRESHOLD:
        return {
            "abstain": True,
            "reason": "low_confidence",
            "message": "Insufficient confidence – manual review recommended.",
            "confidence": round(confidence, 2),
        }
    if modality_mismatch:
        return {
            "abstain": True,
            "reason": "modality_mismatch",
            "message": "Fundus model not yet available; anterior model was used. Manual review recommended.",
            "confidence": round(confidence, 2),
        }

    # -----------------------------
    # Grad-CAM generation
    # -----------------------------
    cam_result = generate_gradcam(image_path)

    original_src = cam_result["original_image_path"]
    gradcam_src  = cam_result["gradcam_image_path"]

    # -----------------------------
    # 🔥 UNIQUE STATIC FILENAMES
    # -----------------------------
    uid = uuid.uuid4().hex

    static_original = os.path.join(STATIC_GEN_DIR, f"{uid}_original.png")
    static_gradcam  = os.path.join(STATIC_GEN_DIR, f"{uid}_gradcam.png")

    shutil.copy(original_src, static_original)
    shutil.copy(gradcam_src, static_gradcam)

    # -----------------------------
    # Return paths RELATIVE to /static
    # -----------------------------
    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "original_image_path": f"generated/{uid}_original.png",
        "gradcam_image_path": f"generated/{uid}_gradcam.png"
    }
