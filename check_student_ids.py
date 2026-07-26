from pathlib import Path

import pandas as pd


processed_folder = Path("data/processed")

csv_files = sorted(
    processed_folder.glob("*.csv")
)

for csv_file in csv_files:
    dataset = pd.read_csv(csv_file)

    print()
    print("=" * 60)
    print(csv_file.name)
    print("Shape:", dataset.shape)
    print("Columns:", dataset.columns.tolist())

    possible_id_columns = [
        "id_student",
        "student_id",
        "studentid",
        "student_number",
        "learner_id",
    ]

    student_id_column = None

    for column in dataset.columns:
        if column.lower().strip() in possible_id_columns:
            student_id_column = column
            break

    if student_id_column is None:
        print("No student ID column was detected.")
        continue

    print("Student ID column:", student_id_column)

    sample_ids = (
        dataset[student_id_column]
        .dropna()
        .drop_duplicates()
        .head(10)
        .tolist()
    )

    print("Sample student IDs:", sample_ids)

    student_id_to_check = 11391

print()
print(f"Checking student {student_id_to_check}")

for csv_file in csv_files:
    dataset = pd.read_csv(csv_file)

    student_id_column = next(
        (
            column
            for column in dataset.columns
            if column.lower().strip()
            in {
                "id_student",
                "student_id",
                "studentid",
                "student_number",
                "learner_id",
            }
        ),
        None,
    )

    if student_id_column is None:
        print(csv_file.name, "- no ID column")
        continue

    found = (
        dataset[student_id_column]
        .astype(str)
        .eq(str(student_id_to_check))
        .any()
    )

    print(
        csv_file.name,
        "- found" if found else "- not found",
    )