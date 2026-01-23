import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from inference import run_inference
from chatbot import get_chatbot_response
from db import init_db, get_db

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

init_db()


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

            inference_result = run_inference(image_path)

            if not inference_result or "error" in inference_result:
                error = inference_result.get("error", "Inference failed.")
            else:
                result = {
                    "prediction": inference_result["prediction"],
                    "confidence": inference_result["confidence"]
                }

                original_image = inference_result["original_image_path"]
                gradcam = inference_result["gradcam_image_path"]

                # ✅ SAVE TO SQLITE (ONE ROW PER INFERENCE)
                conn = get_db()
                c = conn.cursor()
                c.execute("""
                    INSERT INTO history (image_path, gradcam_path, prediction, confidence)
                    VALUES (?, ?, ?, ?)
                """, (
                    original_image,
                    gradcam,
                    result["prediction"],
                    result["confidence"]
                ))
                conn.commit()
                conn.close()

    return render_template(
        "index.html",
        result=result,
        original_image=original_image,
        gradcam=gradcam,
        error=error
    )


@app.route("/history")
def history():
    conn = get_db()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT prediction, confidence, image_path, gradcam_path, timestamp
        FROM history
        ORDER BY timestamp DESC
    """)

    rows = c.fetchall()
    conn.close()

    return render_template("history.html", history=rows)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    result = data.get("result")
    response = get_chatbot_response(result)
    return jsonify({"response": response})


if __name__ == "__main__":
    print("Starting EyeGPT...")
    app.run(debug=True)
