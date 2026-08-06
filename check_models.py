from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys


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
    ".onnx",
    ".tflite",
}

EXPECTED_CHECKPOINTS = (7, 14, 21, 30)

EXPECTED_MODEL_FAMILIES = {
    "Logistic Regression": (
        "logistic",
        "regression",
    ),
    "Decision Tree": (
        "decision",
        "tree",
    ),
    "Random Forest": (
        "random",
        "forest",
    ),
    "XGBoost": (
        "xgboost",
    ),
    "Neural Network": (
        "neural",
        "network",
    ),
}

EXPECTED_NOTEBOOKS = [
    "01_data_exploration.ipynb",
    "02_feature_engineering_and_temporal_datasets.ipynb",
    "03_Logistic_Regression_Full_Course_Baseline.ipynb",
    "04_Logistic_Regression_Temporal_Early_Warning.ipynb",
    "05_Decision_Tree_Temporal_Early_Warning.ipynb",
    "06_Random_Forest_Temporal_Early_Warning.ipynb",
    "07_XGBoost_Temporal_Early_Warning.ipynb",
    "08_Neural_Network_Temporal_Early_Warning.ipynb",
    "09_Model_Comparison_and_Interpretation.ipynb",
    "10_SEAID_Early_Warning_Framework.ipynb",
    "11_SEAID_Explainability_Validation_Deployment.ipynb",
    "12_SEAID_Advanced_Analysis_and_Extensions.ipynb",
]

REQUIRED_DATA_FILES = [
    "data/processed/early_warning_day7_dataset.csv",
    "data/processed/early_warning_day14_dataset.csv",
    "data/processed/early_warning_day21_dataset.csv",
    "data/processed/early_warning_day30_dataset.csv",
]

REQUIRED_RESULT_FILES = [
    "results/all_models_temporal_comparison.csv",
    "results/neural_network_temporal_model_comparison.csv",
    "results/shap_feature_importance_day30.csv",
    "results/shap_behavioral_academic_importance_day30.csv",
    "results/day30_model_summary.csv",
    "results/day30_fairness_metrics.csv",
    "results/day30_fairness_gap_summary.csv",
    "results/day30_decision_confidence_index.csv",
    "results/day30_decision_confidence_summary.csv",
    "results/day30_model_monitoring_baseline.csv",
    "results/day30_prediction_monitoring_baseline.json",
    "results/seaid_monitoring_thresholds.csv",
    "results/seaid_deployment_readiness_checklist.csv",
]

REQUIRED_FIGURE_FILES = [
    "figures/shap_summary_day30.png",
    "figures/shap_global_importance_day30.png",
    "figures/shap_behavioral_academic_day30.png",
    "figures/highest_risk_student_waterfall.png",
    "figures/day30_calibration_curve.png",
    "figures/day30_decision_confidence_distribution.png",
    "figures/day30_accuracy_by_confidence_band.png",
    "figures/temporal_roc_auc_comparison_all_models.png",
]

STREAMLIT_REQUIRED_FILES = [
    "app.py",
    "src/data_loader.py",
    "src/query_engine.py",
    "src/temporal_analysis.py",
]

STREAMLIT_OPTIONAL_FILES = [
    "src/dashboard.py",
    "src/model_comparison.py",
    "src/explainability.py",
    "src/fairness.py",
    "src/confidence.py",
    "src/monitoring.py",
    "src/advisor_dashboard.py",
]

EXCLUDED_TERMS = {
    "checkpoint",
    "epoch",
    "latest",
    "backup",
    "temp",
    "history",
    "optimizer",
}


@dataclass
class ValidationItem:
    label: str
    path: Path
    required: bool = True

    @property
    def exists(self) -> bool:
        return self.path.exists()


def print_section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def find_model_files() -> list[Path]:
    return sorted(
        file_path
        for file_path in PROJECT_ROOT.rglob("*")
        if (
            file_path.is_file()
            and file_path.suffix.lower() in MODEL_EXTENSIONS
        )
    )


def looks_like_family(
    file_path: Path,
    family_terms: tuple[str, ...],
) -> bool:
    searchable_name = str(file_path).lower()
    return all(term in searchable_name for term in family_terms)


def extract_checkpoint(file_path: Path) -> int | None:
    searchable_name = file_path.stem.lower()

    for checkpoint in EXPECTED_CHECKPOINTS:
        patterns = (
            f"day{checkpoint}",
            f"day_{checkpoint}",
            f"day-{checkpoint}",
            f"checkpoint{checkpoint}",
            f"checkpoint_{checkpoint}",
            f"checkpoint-{checkpoint}",
        )

        if any(pattern in searchable_name for pattern in patterns):
            return checkpoint

    return None


def classify_models(
    model_files: list[Path],
) -> dict[str, dict[int | str, list[Path]]]:
    classified: dict[str, dict[int | str, list[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for file_path in model_files:
        matched = False

        for family_name, family_terms in EXPECTED_MODEL_FAMILIES.items():
            if looks_like_family(file_path, family_terms):
                checkpoint = extract_checkpoint(file_path)
                key: int | str = (
                    checkpoint
                    if checkpoint is not None
                    else "Unspecified"
                )
                classified[family_name][key].append(file_path)
                matched = True
                break

        if not matched:
            classified["Unclassified"]["Unspecified"].append(file_path)

    return classified


def possible_final_models(
    model_files: list[Path],
) -> list[Path]:
    final_models = []

    for file_path in model_files:
        searchable_name = str(file_path).lower()

        if any(
            excluded_term in searchable_name
            for excluded_term in EXCLUDED_TERMS
        ):
            continue

        final_models.append(file_path)

    return final_models


def print_model_inventory(model_files: list[Path]) -> None:
    extension_counts = Counter(
        file_path.suffix.lower()
        for file_path in model_files
    )

    folder_counts = Counter(
        relative(file_path.parent)
        for file_path in model_files
    )

    print_section("MODEL FILES BY TYPE")

    if not extension_counts:
        print("No model-like files found.")
    else:
        for extension, count in extension_counts.most_common():
            print(f"{extension:<12} {count:>4}")

    print_section("MODEL FILES BY FOLDER")

    if not folder_counts:
        print("No model-like files found.")
    else:
        for folder, count in folder_counts.most_common():
            print(f"{folder:<55} {count:>4}")


def print_model_family_status(
    classified: dict[str, dict[int | str, list[Path]]],
) -> tuple[int, int]:
    print_section("SEAID MODEL FAMILY AND CHECKPOINT STATUS")

    complete_families = 0
    total_expected = len(EXPECTED_MODEL_FAMILIES)

    for family_name in EXPECTED_MODEL_FAMILIES:
        family_models = classified.get(family_name, {})
        available_checkpoints = {
            checkpoint
            for checkpoint in family_models
            if isinstance(checkpoint, int)
        }

        missing_checkpoints = [
            checkpoint
            for checkpoint in EXPECTED_CHECKPOINTS
            if checkpoint not in available_checkpoints
        ]

        status = "COMPLETE" if not missing_checkpoints else "INCOMPLETE"

        if status == "COMPLETE":
            complete_families += 1

        print()
        print(f"{family_name}: {status}")

        for checkpoint in EXPECTED_CHECKPOINTS:
            paths = family_models.get(checkpoint, [])

            if paths:
                print(
                    f"  [FOUND] Day {checkpoint}: "
                    f"{relative(paths[0])}"
                )

                for duplicate in paths[1:]:
                    print(
                        f"          Additional artifact: "
                        f"{relative(duplicate)}"
                    )
            else:
                print(f"  [MISSING] Day {checkpoint}")

        unspecified = family_models.get("Unspecified", [])

        for path in unspecified:
            print(
                f"  [INFO] Unspecified checkpoint: "
                f"{relative(path)}"
            )

    return complete_families, total_expected


def validate_files(
    title: str,
    relative_paths: list[str],
    required: bool = True,
) -> tuple[int, int]:
    print_section(title)

    items = [
        ValidationItem(
            label=relative_path,
            path=PROJECT_ROOT / relative_path,
            required=required,
        )
        for relative_path in relative_paths
    ]

    found = 0

    for item in items:
        if item.exists:
            found += 1
            print(f"[FOUND]   {item.label}")
        else:
            status = "MISSING" if item.required else "OPTIONAL"
            print(f"[{status:<7}] {item.label}")

    return found, len(items)


def find_notebook(
    expected_name: str,
) -> Path | None:
    notebook_dir = PROJECT_ROOT / "Notebooks"

    exact_path = notebook_dir / expected_name

    if exact_path.exists():
        return exact_path

    normalized_target = (
        expected_name.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    for candidate in notebook_dir.glob("*.ipynb"):
        normalized_candidate = (
            candidate.name.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        if normalized_target in normalized_candidate:
            return candidate

    return None


def validate_notebooks() -> tuple[int, int]:
    print_section("NOTEBOOK STATUS")

    found = 0

    for expected_name in EXPECTED_NOTEBOOKS:
        notebook_path = find_notebook(expected_name)

        if notebook_path is not None:
            found += 1
            print(
                f"[FOUND]   {expected_name} "
                f"-> {relative(notebook_path)}"
            )
        else:
            print(f"[MISSING] {expected_name}")

    return found, len(EXPECTED_NOTEBOOKS)


def print_possible_final_models(
    final_models: list[Path],
) -> None:
    print_section("POSSIBLE FINAL OR DEPLOYABLE MODELS")

    if not final_models:
        print("No possible final models found.")
        return

    for number, model_file in enumerate(
        final_models,
        start=1,
    ):
        size_mb = model_file.stat().st_size / 1_048_576

        print(
            f"{number:03d}. "
            f"{relative(model_file)} "
            f"({size_mb:.2f} MB)"
        )


def print_streamlit_commands() -> None:
    print_section("STREAMLIT STARTUP")

    app_path = PROJECT_ROOT / "app.py"

    if app_path.exists():
        print("Streamlit entry point found.")
        print()
        print("Run from the repository root:")
        print("  python -m streamlit run app.py")
    else:
        print("Streamlit app.py was not found.")


def main() -> int:
    print("=" * 78)
    print("SEAID PROJECT VALIDATION")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")

    model_files = find_model_files()
    classified = classify_models(model_files)
    final_models = possible_final_models(model_files)

    print_model_inventory(model_files)

    complete_families, total_families = (
        print_model_family_status(classified)
    )

    print_possible_final_models(final_models)

    notebooks_found, notebooks_total = validate_notebooks()

    data_found, data_total = validate_files(
        "TEMPORAL DATASET STATUS",
        REQUIRED_DATA_FILES,
    )

    results_found, results_total = validate_files(
        "RESULT FILE STATUS",
        REQUIRED_RESULT_FILES,
    )

    figures_found, figures_total = validate_files(
        "FIGURE FILE STATUS",
        REQUIRED_FIGURE_FILES,
    )

    streamlit_required_found, streamlit_required_total = validate_files(
        "STREAMLIT REQUIRED FILES",
        STREAMLIT_REQUIRED_FILES,
    )

    streamlit_optional_found, streamlit_optional_total = validate_files(
        "STREAMLIT OPTIONAL MODULES",
        STREAMLIT_OPTIONAL_FILES,
        required=False,
    )

    print_streamlit_commands()

    print_section("SEAID VALIDATION SUMMARY")

    summary_rows = [
        (
            "Model families complete",
            complete_families,
            total_families,
        ),
        (
            "Notebooks found",
            notebooks_found,
            notebooks_total,
        ),
        (
            "Temporal datasets found",
            data_found,
            data_total,
        ),
        (
            "Required result files found",
            results_found,
            results_total,
        ),
        (
            "Required figure files found",
            figures_found,
            figures_total,
        ),
        (
            "Required Streamlit files found",
            streamlit_required_found,
            streamlit_required_total,
        ),
        (
            "Optional Streamlit modules found",
            streamlit_optional_found,
            streamlit_optional_total,
        ),
    ]

    for label, found, total in summary_rows:
        print(f"{label:<38} {found:>3} / {total:<3}")

    required_complete = all(
        (
            notebooks_found == notebooks_total,
            data_found == data_total,
            streamlit_required_found == streamlit_required_total,
        )
    )

    research_outputs_complete = all(
        (
            results_found == results_total,
            figures_found == figures_total,
        )
    )

    model_exports_complete = (
        complete_families == total_families
    )

    print()
    print(f"Total model-like files: {len(model_files)}")
    print(f"Possible final models: {len(final_models)}")
    print()

    if required_complete:
        print("[PASS] Core SEAID project structure is ready.")
    else:
        print("[FAIL] Core SEAID project structure is incomplete.")

    if model_exports_complete:
        print(
            "[PASS] All five model families have "
            "Day 7, 14, 21, and 30 artifacts."
        )
    else:
        print(
            "[WARN] One or more model families are missing "
            "checkpoint artifacts."
        )

    if research_outputs_complete:
        print("[PASS] Advanced research outputs are complete.")
    else:
        print(
            "[WARN] Some Notebook 09-12 result or figure "
            "outputs are missing."
        )

    if required_complete:
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
