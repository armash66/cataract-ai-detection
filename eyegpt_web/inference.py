import os
import sys
import shutil

# --------------------------------------------------
# Add PROJECT ROOT to Python path
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Imports (now visible)
# --------------------------------------------------
from anterior_pipeline.src.predict import predict_image
from anterior_pipeline.src.gradcam import generate_gradcam
from quality import check_image_quality


def run_inference(image_path):
    ok, message = check_image_quality(image_path)
    if not ok:
        return {"error": message}

    label, confidence = predict_image(image_path)

    gradcam_path = generate_gradcam(image_path)
    static_path = os.path.join(BASE_DIR, "static", "gradcam_result.png")
    shutil.copy(gradcam_path, static_path)

    return {
        "prediction": label,
        "confidence": confidence,
        "gradcam": "gradcam_result.png"
    }
