from pathlib import Path

import pandas as pd


output_directory = Path("data/processed")
output_directory.mkdir(parents=True, exist_ok=True)

test_data = pd.DataFrame(
    {
        "id_student": [
            1001,
            1002,
            1003,
            1004,
            1005,
            1006,
            1007,
            1008,
        ],
        "gender": [
            "F",
            "M",
            "F",
            "M",
            "F",
            "M",
            "F",
            "M",
        ],
        "highest_education": [
            "A Level",
            "Lower Than A Level",
            "HE Qualification",
            "A Level",
            "A Level",
            "Lower Than A Level",
            "HE Qualification",
            "A Level",
        ],
        "studied_credits": [
            60,
            60,
            120,
            60,
            90,
            60,
            120,
            90,
        ],
        "num_of_prev_attempts": [
            0,
            1,
            0,
            2,
            0,
            1,
            0,
            1,
        ],
        "total_vle_clicks": [
            420,
            115,
            780,
            90,
            510,
            210,
            850,
            305,
        ],
        "checkpoint_day": [
            14,
            14,
            14,
            14,
            14,
            14,
            14,
            14,
        ],
        "student_success": [
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            1,
        ],
    }
)

output_path = (
    output_directory
    / "seaid_test_data.csv"
)

test_data.to_csv(
    output_path,
    index=False,
)

print(f"Saved test dataset to: {output_path}")
print(f"Dataset shape: {test_data.shape}")