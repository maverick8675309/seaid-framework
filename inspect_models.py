from pathlib import Path

import joblib
from xgboost import XGBClassifier


MODEL_ROOT = Path("models")
CHECKPOINTS = [7, 14, 21, 30]


print("=" * 70)
print("SEAID PORTABLE XGBOOST MODEL INSPECTION")
print("=" * 70)

loaded_models = {}

for checkpoint_day in CHECKPOINTS:

    day_dir = MODEL_ROOT / f"day_{checkpoint_day:02d}"

    preprocessing_path = (
        day_dir / f"preprocessing_day{checkpoint_day}.joblib"
    )

    model_path = (
        day_dir / f"xgboost_day{checkpoint_day}.json"
    )

    print(f"\nCheckpoint: Day {checkpoint_day}")
    print(f"Preprocessing: {preprocessing_path}")
    print(f"Model:         {model_path}")

    try:
        preprocessing = joblib.load(preprocessing_path)

        model = XGBClassifier()
        model.load_model(model_path)

        loaded_models[checkpoint_day] = {
            "preprocessing": preprocessing,
            "model": model,
        }

        print("Successfully loaded")
        print(
            f"Preprocessing type: "
            f"{type(preprocessing).__name__}"
        )
        print(f"Model type: {type(model).__name__}")
        print(f"Features: {model.n_features_in_}")

    except Exception as error:
        print("Failed to load")
        print(f"{type(error).__name__}: {error}")


print("\n" + "=" * 70)
print(
    f"Successfully loaded "
    f"{len(loaded_models)} of {len(CHECKPOINTS)} models."
)
print("=" * 70)