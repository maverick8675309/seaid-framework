from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def find_processed_csv_files() -> list[Path]:
    """
    Return all CSV files in data/processed.
    """

    if not PROCESSED_DATA_DIRECTORY.exists():
        return []

    return sorted(
        PROCESSED_DATA_DIRECTORY.glob(
            "*.csv"
        )
    )


@st.cache_data
def load_processed_dataset(
    file_path: str,
) -> pd.DataFrame:
    """
    Load and validate one processed CSV.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError(
            "The selected dataset contains no records."
        )

    return dataset


@st.cache_data
def load_temporal_datasets(
    checkpoint_paths: dict[int, str],
) -> dict[int, pd.DataFrame]:
    """
    Load all checkpoint datasets.
    """

    temporal_datasets: dict[
        int,
        pd.DataFrame,
    ] = {}

    for checkpoint_day, file_path in (
        checkpoint_paths.items()
    ):
        temporal_datasets[
            checkpoint_day
        ] = pd.read_csv(file_path)

    return temporal_datasets


def summarize_dataset(
    dataset: pd.DataFrame,
) -> dict[str, object]:
    """
    Calculate general dataset information.
    """

    return {
        "rows": int(dataset.shape[0]),
        "columns": int(
            dataset.shape[1]
        ),
        "column_names": (
            dataset.columns.tolist()
        ),
        "duplicate_rows": int(
            dataset.duplicated().sum()
        ),
        "missing_values": int(
            dataset.isna().sum().sum()
        ),
    }


def find_student_id_column(
    dataset: pd.DataFrame,
) -> str | None:
    """
    Identify a likely student ID column.
    """

    possible_names = [
        "id_student",
        "student_id",
        "studentid",
        "student_number",
        "learner_id",
    ]

    normalized_columns = {
        column.lower().strip(): column
        for column in dataset.columns
    }

    for name in possible_names:
        if name in normalized_columns:
            return normalized_columns[
                name
            ]

    return None


def find_target_column(
    dataset: pd.DataFrame,
) -> str | None:
    """
    Identify a likely target column.
    """

    possible_names = [
        "target",
        "success",
        "student_success",
        "final_result",
        "outcome",
        "label",
    ]

    normalized_columns = {
        column.lower().strip(): column
        for column in dataset.columns
    }

    for name in possible_names:
        if name in normalized_columns:
            return normalized_columns[
                name
            ]

    return None