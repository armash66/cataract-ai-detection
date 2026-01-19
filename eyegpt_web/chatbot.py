def get_chatbot_response(result):
    """
    Rule-based EyeGPT explanation.
    No external APIs.
    Safe for medical AI demos.
    """

    if not result:
        return (
            "Please analyze an eye image first. "
            "Once a screening result is available, EyeGPT can explain it."
        )

    prediction = result.get("prediction", "Unknown")
    confidence = result.get("confidence", 0)

    explanation = []

    explanation.append("🧠 **EyeGPT Screening Explanation**\n")

    explanation.append(
        "The uploaded eye image was analyzed using a deep learning model "
        "trained on anterior (front-facing) eye images."
    )

    if prediction.lower() == "cataract":
        explanation.append(
            f"\n🔍 The model detected visual patterns commonly associated with **cataract**."
        )
    else:
        explanation.append(
            f"\n🔍 The model did **not** detect strong cataract-related patterns."
        )

    explanation.append(
        f"\n📊 **Confidence Score:** {confidence}%"
    )

    if confidence >= 85:
        explanation.append(
            "\nThis is a high confidence result, meaning the model found strong similarity "
            "with images it has previously learned from."
        )
    elif confidence >= 60:
        explanation.append(
            "\nThis is a moderate confidence result. The image shares some features "
            "with known examples, but uncertainty remains."
        )
    else:
        explanation.append(
            "\nThis is a low confidence result. Lighting, focus, or image quality "
            "may have affected the analysis."
        )

    explanation.append(
        "\n\n⚠️ **Important Disclaimer:**\n"
        "This result is intended for educational and research purposes only. "
        "It is not a medical diagnosis and must not replace professional "
        "ophthalmic evaluation."
    )

    return "\n".join(explanation)
