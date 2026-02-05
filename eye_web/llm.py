def generate_explanation(result):
    pred = result["prediction"]
    conf = result["confidence"]

    return (
        f"The system detected **{pred}** with a confidence of {conf:.1f}%. "
        "This assessment is based on visual patterns observed in the anterior eye image. "
        "This tool is intended for screening and educational purposes only and "
        "should not be considered a medical diagnosis. "
        "For accurate diagnosis and treatment, please consult a qualified ophthalmologist."
    )
