import pandas as pd


def predict_student_risk(
    student_data: pd.DataFrame,
    preprocessing,
    model,
) -> dict:
    if student_data.empty:
        raise ValueError("Student data is empty.")

    if len(student_data) != 1:
        raise ValueError(
            "Prediction currently requires exactly one student record."
        )

    transformed_features = preprocessing.transform(
        student_data
    )

    risk_probability = float(
        model.predict_proba(
            transformed_features
        )[0, 1]
    )

    predicted_class = int(
        risk_probability >= 0.50
    )

    if risk_probability >= 0.70:
        risk_level = "High"
    elif risk_probability >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    return {
        "probability": risk_probability,
        "predicted_class": predicted_class,
        "risk_level": risk_level,
    }