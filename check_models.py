from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_EXTENSIONS = {
    ".joblib",
    ".pkl",
    ".pickle",
    ".json",
    ".ubj",
    ".keras",
    ".h5",
    ".pt",
    ".pth",
}


model_files = sorted(
    file_path
    for file_path in PROJECT_ROOT.rglob("*")
    if (
        file_path.is_file()
        and file_path.suffix.lower()
        in MODEL_EXTENSIONS
    )
)


extension_counts = Counter(
    file_path.suffix.lower()
    for file_path in model_files
)

folder_counts = Counter(
    str(file_path.parent.relative_to(PROJECT_ROOT))
    for file_path in model_files
)


print("=" * 70)
print("MODEL FILES BY TYPE")
print("=" * 70)

for extension, count in extension_counts.most_common():
    print(f"{extension}: {count}")


print()
print("=" * 70)
print("MODEL FILES BY FOLDER")
print("=" * 70)

for folder, count in folder_counts.most_common():
    print(f"{folder}: {count}")


print()
print("=" * 70)
print("POSSIBLE FINAL MODELS")
print("=" * 70)


excluded_terms = {
    "checkpoint",
    "epoch",
    "latest",
    "backup",
    "temp",
    "history",
    "optimizer",
}


possible_final_models = []

for file_path in model_files:
    searchable_name = str(file_path).lower()

    if any(
        excluded_term in searchable_name
        for excluded_term in excluded_terms
    ):
        continue

    possible_final_models.append(file_path)


for number, model_file in enumerate(
    possible_final_models,
    start=1,
):
    relative_path = model_file.relative_to(
        PROJECT_ROOT
    )

    size_mb = (
        model_file.stat().st_size
        / 1_048_576
    )

    print(
        f"{number:03d}. "
        f"{relative_path} "
        f"({size_mb:.2f} MB)"
    )


print()
print(
    "Total model-like files:",
    len(model_files),
)

print(
    "Possible final models:",
    len(possible_final_models),
)