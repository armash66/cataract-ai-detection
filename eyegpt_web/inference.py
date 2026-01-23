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


import uuid

def run_inference(image_path):
    """
    Full inference pipeline:
    - Quality check
    - Prediction
    - Grad-CAM generation

    Returns:
        prediction
        confidence
        original_image_path
        gradcam_image_path
    """

    # -----------------------------
    # Quality gate
    # -----------------------------
    ok, message = check_image_quality(image_path)
    if not ok:
        return {"error": message}

    # -----------------------------
    # Prediction
    # -----------------------------
    label, confidence = predict_image(image_path)

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
