from flask import Flask, render_template, request
import os
from inference import run_inference
from llm import generate_explanation

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = explanation = gradcam = error = None

    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        result = run_inference(image_path)

        if "error" in result:
            error = result["error"]
            result = None
        else:
            explanation = generate_explanation(result)
            gradcam = result.get("gradcam")

    return render_template(
        "index.html",
        result=result,
        explanation=explanation,
        gradcam=gradcam,
        error=error
    )

if __name__ == "__main__":
    print("Starting EyeGPT server...")
    app.run(debug=True)
