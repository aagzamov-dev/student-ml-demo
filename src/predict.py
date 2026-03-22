from src.config import PASS_LABELS


def predict_student_outcome(model_bundle, input_df):
    """Predict whether a student is likely to pass."""
    X_scaled = model_bundle["scaler"].transform(input_df)
    pass_probability = float(model_bundle["classifier"].predict_proba(X_scaled)[0][1])
    predicted_class = int(pass_probability >= 0.5)

    return {
        "prediction": PASS_LABELS[predicted_class],
        "pass_probability": round(pass_probability * 100, 2),
        "risk_probability": round((1 - pass_probability) * 100, 2),
        "predicted_class": predicted_class,
    }
