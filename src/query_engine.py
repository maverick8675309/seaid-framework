from __future__ import annotations

import re
from typing import Any, Final

import pandas as pd

from src.student_analysis import (
    find_student_records,
    format_student_record,
)

from src.temporal_analysis import (
    compare_student_checkpoints,
)


# -------------------------------------------------------
# SEAID Knowledge Base
# -------------------------------------------------------

MODEL_PERFORMANCE: Final[dict[str, float]] = {
    "XGBoost": 0.8614,
    "Neural Network": 0.8571,
    "Logistic Regression": 0.8264,
    "Random Forest": 0.8202,
}

FEATURES: Final[list[str]] = [
    "Gender",
    "Highest Education",
    "IMD Band",
    "Age Band",
    "Number of Previous Attempts",
    "Studied Credits",
    "Disability Status",
    "Total VLE Clicks",
]

TARGET_VARIABLE: Final[str] = (
    "Student Success "
    "(Pass or Distinction = 1; Fail or Withdrawn = 0)"
)


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def normalize_query(query: str) -> str:
    """
    Convert a user query to normalized lowercase text.
    """

    return re.sub(
        r"\s+",
        " ",
        query.lower().strip(),
    )


def format_model_performance() -> str:
    """
    Return model performance ranked from highest to lowest ROC-AUC.
    """

    ranked_models = sorted(
        MODEL_PERFORMANCE.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    lines = [
        "## Current SEAID Model Performance",
        "",
    ]

    for rank, (model_name, score) in enumerate(
        ranked_models,
        start=1,
    ):
        lines.append(
            f"{rank}. **{model_name}** — ROC-AUC: **{score:.4f}**"
        )

    return "\n".join(lines)


def get_best_model() -> tuple[str, float]:
    """
    Return the best-performing model and its ROC-AUC score.
    """

    model_name = max(
        MODEL_PERFORMANCE,
        key=MODEL_PERFORMANCE.get,
    )

    return model_name, MODEL_PERFORMANCE[model_name]


def identify_requested_model(
    query: str,
) -> str | None:
    """
    Identify a model mentioned in a user query.
    """

    model_aliases = {
        "xgboost": "XGBoost",
        "xg boost": "XGBoost",
        "neural network": "Neural Network",
        "deep learning model": "Neural Network",
        "logistic regression": "Logistic Regression",
        "logistic model": "Logistic Regression",
        "random forest": "Random Forest",
        "forest model": "Random Forest",
    }

    for alias, model_name in model_aliases.items():
        if alias in query:
            return model_name

    return None


def format_feature_list() -> str:
    """
    Return the current SEAID feature list.
    """

    feature_lines = [
        f"- {feature}"
        for feature in FEATURES
    ]

    return (
        "The current SEAID predictive variables include:\n\n"
        + "\n".join(feature_lines)
    )


def format_dataset_columns(
    column_names: list[str],
) -> str:
    """
    Format dataset column names as a Markdown list.
    """

    columns = "\n".join(
        f"- `{column}`"
        for column in column_names
    )

    return (
        "The loaded dataset contains these columns:\n\n"
        f"{columns}"
    )


def dataset_is_loaded(
    dataset: pd.DataFrame | None,
) -> bool:
    """
    Return True when a usable dataset is available.
    """

    return (
        dataset is not None
        and isinstance(dataset, pd.DataFrame)
        and not dataset.empty
    )


def extract_student_id(
    query: str,
) -> str | None:
    """
    Extract a numeric student ID from a natural-language query.

    Examples:
        Analyze student 1003
        Show student 1003
        Find student ID 1003
    """

    patterns = [
        r"student\s+id\s*[:#-]?\s*(\d+)",
        r"student\s*[:#-]?\s*(\d+)",
        r"id\s*[:#-]?\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None


# -------------------------------------------------------
# Main Query Engine
# -------------------------------------------------------

def answer_query(
    query: str,
    dataset: pd.DataFrame | None = None,
    dataset_summary: dict[str, Any] | None = None,
    student_id_column: str | None = None,
    target_column: str | None = None,
    temporal_datasets: dict[
        int,
        pd.DataFrame,
    ] | None = None,
    student_id_columns: dict[
        int,
        str | None,
    ] | None = None,
    target_columns: dict[
        int,
        str | None,
    ] | None = None,
) -> str:
    """
    Answer questions using SEAID project knowledge and a loaded dataset.

    Parameters
    ----------
    query:
        The user's question.

    dataset:
        The currently loaded processed dataset.

    dataset_summary:
        Summary information created by data_loader.py.

    student_id_column:
        Automatically detected student identifier column.

    target_column:
        Automatically detected prediction target column.

    Returns
    -------
    str
        A Markdown-formatted response.
    """

    normalized_query = normalize_query(query)

    if not normalized_query:
        return "Please enter a question about SEAID."

    # ---------------------------------------------------
    # Greetings
    # ---------------------------------------------------

    if normalized_query in {
        "hello",
        "hi",
        "hey",
        "hello seaid",
        "hi seaid",
        "hey seaid",
    }:
        return (
            "Hello! I am the SEAID assistant.\n\n"
            "You can ask me about:\n\n"
            "- The SEAID framework\n"
            "- Model performance\n"
            "- Predictive variables\n"
            "- The OULAD dataset\n"
            "- Explainability\n"
            "- Responsible AI\n"
            "- The currently loaded processed dataset\n"
            "- Individual student records"
        )
    # ---------------------------------------------------
    # Student Comparison Across Checkpoints
    # ---------------------------------------------------

    requested_student_id = extract_student_id(
        normalized_query
    )

    if (
        requested_student_id is not None
        and any(
            phrase in normalized_query
            for phrase in [
                "across checkpoints",
                "compare student",
                "over time",
                "temporal comparison",
                "across time",
                "all checkpoints",
            ]
        )
    ):
        return compare_student_checkpoints(
            student_id=requested_student_id,
            temporal_datasets=(
                temporal_datasets or {}
            ),
            student_id_columns=(
                student_id_columns or {}
            ),
            target_columns=(
                target_columns or {}
            ),
        )
    # ---------------------------------------------------
    # Live Individual Student Lookup
    # ---------------------------------------------------

    requested_student_id = extract_student_id(
        normalized_query
    )

    if (
        requested_student_id is not None
        and any(
            phrase in normalized_query
            for phrase in [
                "student",
                "analyze",
                "show",
                "find",
                "lookup",
                "review",
            ]
        )
    ):
        if not dataset_is_loaded(dataset):
            return (
                "No processed dataset is currently loaded."
            )

        if student_id_column is None:
            return (
                "I could not identify the student ID column "
                "in the loaded dataset."
            )

        try:
            student_records = find_student_records(
                dataset=dataset,
                student_id_column=student_id_column,
                student_id=requested_student_id,
            )

            return format_student_record(
                student_records=student_records,
                student_id=requested_student_id,
                student_id_column=student_id_column,
                target_column=target_column,
            )

        except ValueError:
            return (
                f"Student ID **{requested_student_id}** "
                "could not be converted to the format used "
                "by the dataset."
            )

        except Exception as error:
            return (
                "SEAID encountered an error while retrieving "
                f"the student record: `{error}`"
            )

    # ---------------------------------------------------
    # Live Dataset: Number of Records
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "how many records",
            "how many rows",
            "number of records",
            "number of rows",
            "dataset size",
            "size of the dataset",
        ]
    ):
        if dataset_summary is None:
            return (
                "No processed dataset is currently loaded. "
                "Place a CSV file inside `data/processed/`."
            )

        return (
            "The currently loaded dataset contains "
            f"**{dataset_summary['rows']:,} records**."
        )

    # ---------------------------------------------------
    # Live Dataset: Number of Columns
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "how many columns",
            "number of columns",
            "how many variables",
            "number of variables",
        ]
    ):
        if dataset_summary is None:
            return (
                "No processed dataset is currently loaded."
            )

        return (
            "The currently loaded dataset contains "
            f"**{dataset_summary['columns']:,} columns**."
        )

    # ---------------------------------------------------
    # Live Dataset: Column Names
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "what columns",
            "list columns",
            "column names",
            "show columns",
            "variables in the dataset",
            "what variables are loaded",
        ]
    ):
        if dataset_summary is None:
            return (
                "No processed dataset is currently loaded."
            )

        column_names = dataset_summary.get(
            "column_names",
            [],
        )

        return format_dataset_columns(
            column_names
        )

    # ---------------------------------------------------
    # Live Dataset: Missing Values
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "missing values",
            "missing data",
            "how many missing",
            "null values",
            "null data",
            "na values",
        ]
    ):
        if dataset_summary is None:
            return (
                "No processed dataset is currently loaded."
            )

        missing_values = dataset_summary.get(
            "missing_values",
            0,
        )

        if missing_values == 0:
            return (
                "The loaded dataset contains "
                "**no missing values**."
            )

        return (
            "The loaded dataset contains "
            f"**{missing_values:,} missing values** "
            "across all rows and columns."
        )

    # ---------------------------------------------------
    # Live Dataset: Duplicate Rows
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "duplicate rows",
            "duplicates",
            "duplicated records",
            "duplicate records",
            "are there duplicates",
        ]
    ):
        if dataset_summary is None:
            return (
                "No processed dataset is currently loaded."
            )

        duplicate_rows = dataset_summary.get(
            "duplicate_rows",
            0,
        )

        if duplicate_rows == 0:
            return (
                "The loaded dataset contains "
                "**no duplicate rows**."
            )

        return (
            "The loaded dataset contains "
            f"**{duplicate_rows:,} duplicate rows**."
        )

    # ---------------------------------------------------
    # Live Dataset: Student ID Column
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "student id column",
            "student identifier",
            "id variable",
            "student id variable",
            "which column identifies students",
        ]
    ):
        if not dataset_is_loaded(dataset):
            return (
                "No processed dataset is currently loaded."
            )

        if student_id_column is None:
            return (
                "I could not automatically identify a student "
                "ID column in the loaded dataset."
            )

        return (
            "The detected student identifier column is "
            f"**`{student_id_column}`**."
        )

    # ---------------------------------------------------
    # Live Dataset: Target Column
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "target column",
            "target variable in the dataset",
            "outcome column",
            "label column",
            "which column is the target",
        ]
    ):
        if not dataset_is_loaded(dataset):
            return (
                "No processed dataset is currently loaded."
            )

        if target_column is None:
            return (
                "I could not automatically identify a target "
                "column in the loaded dataset."
            )

        return (
            "The detected target column is "
            f"**`{target_column}`**."
        )

    # ---------------------------------------------------
    # Live Dataset: Unique Students
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "how many students",
            "number of students",
            "unique students",
            "student count",
        ]
    ):
        if not dataset_is_loaded(dataset):
            return (
                "No processed dataset is currently loaded."
            )

        if student_id_column is None:
            return (
                "I could not identify the student ID column, "
                "so I cannot calculate the number of unique students."
            )

        unique_students = int(
            dataset[student_id_column].nunique()
        )

        return (
            "The loaded dataset contains "
            f"**{unique_students:,} unique students**."
        )

    # ---------------------------------------------------
    # Live Dataset: Target Distribution
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "target distribution",
            "outcome distribution",
            "class distribution",
            "show target values",
            "count outcomes",
            "show the target distribution",
        ]
    ):
        if not dataset_is_loaded(dataset):
            return (
                "No processed dataset is currently loaded."
            )

        if target_column is None:
            return (
                "I could not identify the target column in "
                "the loaded dataset."
            )

        value_counts = (
            dataset[target_column]
            .value_counts(dropna=False)
        )

        lines = [
            f"- **{value}**: {count:,}"
            for value, count in value_counts.items()
        ]

        return (
            f"The distribution of **`{target_column}`** is:\n\n"
            + "\n".join(lines)
        )

    # ---------------------------------------------------
    # SEAID Definition
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "what is seaid",
            "tell me about seaid",
            "describe seaid",
            "explain seaid",
            "purpose of seaid",
        ]
    ) or normalized_query == "seaid":
        return (
            "SEAID stands for **Student Explainable Artificial "
            "Intelligence and Deep Learning Framework for "
            "Educational Decision Support**.\n\n"
            "The framework combines:\n\n"
            "- Machine learning\n"
            "- Deep learning\n"
            "- Explainable artificial intelligence\n"
            "- Learning analytics\n"
            "- Temporal prediction\n"
            "- Human-centered educational decision support\n\n"
            "Its purpose is to identify patterns associated with "
            "student success while providing educators with "
            "transparent and responsible decision-support information."
        )

    # ---------------------------------------------------
    # Best Model
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "best model",
            "top model",
            "highest performing model",
            "strongest model",
            "most accurate model",
            "highest auc",
            "highest roc",
            "which model performs best",
        ]
    ):
        model_name, score = get_best_model()

        return (
            "The current best-performing SEAID model is "
            f"**{model_name}**, with a ROC-AUC of "
            f"**{score:.4f}**."
        )

    # ---------------------------------------------------
    # Individual Model Performance
    # ---------------------------------------------------

    requested_model = identify_requested_model(
        normalized_query
    )

    if (
        requested_model is not None
        and any(
            term in normalized_query
            for term in [
                "score",
                "performance",
                "auc",
                "roc",
                "result",
                "how did",
            ]
        )
    ):
        score = MODEL_PERFORMANCE[
            requested_model
        ]

        return (
            f"**{requested_model}** currently has a "
            f"ROC-AUC score of **{score:.4f}**."
        )

    # ---------------------------------------------------
    # Compare Models
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "compare models",
            "compare all models",
            "all models",
            "model comparison",
            "model performance",
            "performance rankings",
            "roc-auc scores",
            "roc auc scores",
        ]
    ):
        return format_model_performance()

    # ---------------------------------------------------
    # Models Included
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "which models",
            "what models",
            "models included",
            "implemented models",
            "algorithms included",
            "classifiers included",
        ]
    ):
        model_names = "\n".join(
            f"- {model_name}"
            for model_name in MODEL_PERFORMANCE
        )

        return (
            "SEAID currently includes these models:\n\n"
            f"{model_names}"
        )

    # ---------------------------------------------------
    # Predictive Features
    # ---------------------------------------------------

    if any(
        word in normalized_query
        for word in [
            "feature",
            "features",
            "variable",
            "variables",
            "predictor",
            "predictors",
            "input",
            "inputs",
            "factor",
            "factors",
        ]
    ):
        return format_feature_list()

    # ---------------------------------------------------
    # Project Dataset
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "what dataset",
            "which dataset",
            "data source",
            "oulad",
            "open university dataset",
        ]
    ):
        return (
            "SEAID currently uses the **Open University Learning "
            "Analytics Dataset (OULAD)** as its proof-of-concept "
            "data source.\n\n"
            "OULAD includes:\n\n"
            "- Student demographic information\n"
            "- Registration information\n"
            "- Assessment activity and performance\n"
            "- Virtual Learning Environment activity\n"
            "- Course information\n"
            "- Final student outcomes"
        )

    # ---------------------------------------------------
    # Project Target Variable
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "what does seaid predict",
            "what is the target",
            "target variable",
            "outcome variable",
            "success definition",
            "define student success",
        ]
    ):
        return (
            "SEAID currently predicts:\n\n"
            f"**{TARGET_VARIABLE}**"
        )

    # ---------------------------------------------------
    # Explainability
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "explainability",
            "explainable",
            "explanation",
            "shap",
            "feature importance",
            "why prediction",
            "explain prediction",
        ]
    ):
        return (
            "SEAID's explainability layer is designed to use "
            "**SHAP values, feature importance, and local "
            "student-level explanations**.\n\n"
            "These methods help educators understand which "
            "variables influenced a model prediction rather "
            "than receiving only an unexplained risk score."
        )

    # ---------------------------------------------------
    # Ethics and Responsible AI
    # ---------------------------------------------------

    if any(
        word in normalized_query
        for word in [
            "ethics",
            "ethical",
            "fairness",
            "bias",
            "responsible",
            "equity",
            "oversight",
            "privacy",
        ]
    ):
        return (
            "SEAID is designed as a **human-in-the-loop "
            "decision-support system**, not an automated "
            "decision-maker.\n\n"
            "Its responsible-use principles include:\n\n"
            "- Human oversight\n"
            "- Transparency\n"
            "- Explainability\n"
            "- Fairness monitoring\n"
            "- Educational equity\n"
            "- Privacy protection\n"
            "- Careful interpretation of model outputs"
        )

    # ---------------------------------------------------
    # General Student Analysis
    # ---------------------------------------------------

    if any(
        phrase in normalized_query
        for phrase in [
            "analyze student",
            "predict student",
            "student risk",
            "student prediction",
            "individual student",
        ]
    ):
        return (
            "To review a student from the loaded dataset, "
            "include the student ID in your question.\n\n"
            "For example:\n\n"
            "- `Analyze student 1003`\n"
            "- `Show student 1006`\n"
            "- `Find student ID 1008`\n\n"
            "The current version retrieves recorded student data. "
            "A trained-model prediction will be added in a later step."
        )

    # ---------------------------------------------------
    # Broad SEAID Match
    # ---------------------------------------------------

    if "seaid" in normalized_query:
        return (
            "**SEAID** is an explainable artificial intelligence "
            "and deep learning framework designed to support "
            "student-success decisions through transparent, "
            "human-centered predictive analytics."
        )

    # ---------------------------------------------------
    # Broad Model Match
    # ---------------------------------------------------

    if any(
        word in normalized_query
        for word in [
            "model",
            "models",
            "algorithm",
            "algorithms",
            "classifier",
            "classifiers",
        ]
    ):
        return format_model_performance()

    # ---------------------------------------------------
    # Default Response
    # ---------------------------------------------------

    return (
        "I could not match that question to the current SEAID "
        "knowledge base.\n\n"
        "Try asking:\n\n"
        "- What is SEAID?\n"
        "- Which model performs best?\n"
        "- Compare all models.\n"
        "- What dataset does SEAID use?\n"
        "- Which features are included?\n"
        "- How many records are loaded?\n"
        "- How many students are in the dataset?\n"
        "- Are there missing values?\n"
        "- What is the target column?\n"
        "- Show the target distribution.\n"
        "- Analyze student 1003.\n"
        "- How does SEAID explain predictions?\n"
        "- How does SEAID support responsible AI?"
    )