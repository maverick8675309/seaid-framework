from pathlib import Path

import joblib
from xgboost import XGBClassifier


MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"


def load_checkpoint_model(checkpoint_day: int):
    valid_checkpoints = {7, 14, 21, 30}

    if checkpoint_day not in valid_checkpoints:
        raise ValueError(
            f"Checkpoint day must be one of "
            f"{sorted(valid_checkpoints)}."
        )

    day_directory = MODEL_ROOT / f"day_{checkpoint_day:02d}"

    preprocessing_path = (
        day_directory
        / f"preprocessing_day{checkpoint_day}.joblib"
    )

    model_path = (
        day_directory
        / f"xgboost_day{checkpoint_day}.json"
    )

    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: "
            f"{preprocessing_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {model_path}"
        )

    preprocessing = joblib.load(preprocessing_path)

    model = XGBClassifier()
    model.load_model(model_path)

    return preprocessing, model