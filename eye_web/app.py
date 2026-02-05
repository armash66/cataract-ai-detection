import sqlite3
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from inference import run_inference
from chatbot import get_chatbot_response
from db import init_db, get_db
import uuid

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
            ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4().hex}{ext}"
            image_path = os.path.join(UPLOAD_FOLDER, unique_name)
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
        SELECT id, image_path, gradcam_path, prediction, confidence, timestamp
        FROM history
        ORDER BY timestamp DESC
    """)

    rows = c.fetchall()
    conn.close()

    return render_template("history.html", history=rows)


@app.route("/protocols")
def protocols():
    return render_template("protocols.html")


@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/history/clear", methods=["POST"])
def clear_history():
    conn = get_db()
    c = conn.cursor()

    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()

    return redirect(url_for("history"))


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    result = data.get("result")
    response = get_chatbot_response(result)
    return jsonify({"response": response})

@app.route("/delete/<int:id>", methods=["POST"])
def delete_history(id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id = ?", (id,))
    conn.commit()
    conn.close()
    return redirect(url_for("history"))

if __name__ == "__main__":
    print("Starting EyeGPT...")
    app.run(debug=True)
