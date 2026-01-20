from flask import Flask, render_template, request, jsonify
import os

from inference import run_inference
from chatbot import get_chatbot_response

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    gradcam = None
    original_image = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Run model inference (now returns structured output)
            inference_result = run_inference(image_path)

            if not inference_result or "error" in inference_result:
                error = inference_result.get("error", "Inference failed.")
                result = None
            else:
                # Core prediction info (unchanged contract for UI)
                result = {
                    "prediction": inference_result.get("prediction"),
                    "confidence": inference_result.get("confidence")
                }

                # NEW: explicit images
                original_image = inference_result.get("original_image_path")
                gradcam = inference_result.get("gradcam_image_path")

    return render_template(
        "index.html",
        result=result,
        original_image=original_image,
        gradcam=gradcam,
        error=error
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    result = data.get("result")
    response = get_chatbot_response(result)
    return jsonify({"response": response})


if __name__ == "__main__":
    print("Starting EyeGPT (Rule-Based Mode)...")
    app.run(debug=True)
