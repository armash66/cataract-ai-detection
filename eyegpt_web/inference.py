import sys
import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from anterior_pipeline.src.predict import predict_image
from anterior_pipeline.src.gradcam import generate_gradcam


def run_inference(image_path):
    label, confidence = predict_image(image_path)

    gradcam_path = generate_gradcam(image_path)

    # Copy gradcam image to static folder
    static_gradcam = os.path.join("static", "gradcam_result.png")
    shutil.copy(gradcam_path, static_gradcam)

    return {
        "prediction": label,
        "confidence": confidence,
        "gradcam": "gradcam_result.png"
    }
