import pandas as pd

df = pd.read_csv(
    "data/processed/early_warning_day14_dataset.csv"
)

df.head(1).to_csv(
    "test_student.csv",
    index=False
)

print("Created test_student.csv")