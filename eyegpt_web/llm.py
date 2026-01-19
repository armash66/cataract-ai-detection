def generate_explanation(result):
    pred = result["prediction"]
    conf = result["confidence"]

    return (
        f"The system detected **{pred}** with a confidence of {conf:.2f}%. "
        "This result is based on analysis of the uploaded anterior eye image. "
        "This is a screening result, not a medical diagnosis. "
        "Please consult an ophthalmologist for professional evaluation."
    )
