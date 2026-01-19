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
    error = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            result = run_inference(image_path)

            if "error" in result:
                error = result["error"]
                result = None
            else:
                gradcam = result.get("gradcam")

    return render_template(
        "index.html",
        result=result,
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
