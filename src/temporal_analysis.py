from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def extract_checkpoint_day(
    file_path: str | Path,
) -> int | None:
    """
    Extract a checkpoint day from a filename.

    Examples:
        seaid_day_07.csv -> 7
        seaid_day_14.csv -> 14
        day_30.csv -> 30
    """

    filename = Path(file_path).stem.lower()

    patterns = [
        r"day[_\s-]?(\d+)",
        r"checkpoint[_\s-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)

        if match:
            return int(match.group(1))

    return None


def build_checkpoint_map(
    file_paths: list[Path],
) -> dict[int, Path]:
    """
    Create a mapping of checkpoint day to CSV path.
    """

    checkpoint_map: dict[int, Path] = {}

    for file_path in file_paths:
        checkpoint_day = extract_checkpoint_day(
            file_path
        )

        if checkpoint_day is not None:
            checkpoint_map[checkpoint_day] = file_path

    return dict(
        sorted(checkpoint_map.items())
    )


def normalize_student_id(
    student_id: str,
    dataset: pd.DataFrame,
    student_id_column: str,
) -> Any:
    """
    Convert a student ID into the data type used
    by the dataset's student ID column.
    """

    student_series = dataset[
        student_id_column
    ]

    if pd.api.types.is_integer_dtype(
        student_series
    ):
        return int(student_id)

    if pd.api.types.is_float_dtype(
        student_series
    ):
        return float(student_id)

    return str(student_id)


def find_student_at_checkpoint(
    dataset: pd.DataFrame,
    student_id_column: str,
    student_id: str,
) -> pd.DataFrame:
    """
    Retrieve all rows for one student from
    a checkpoint dataset.
    """

    normalized_student_id = normalize_student_id(
        student_id=student_id,
        dataset=dataset,
        student_id_column=student_id_column,
    )

    return dataset[
        dataset[student_id_column]
        == normalized_student_id
    ].copy()


def format_value(
    value: Any,
) -> str:
    """
    Format values for Markdown display.
    """

    if pd.isna(value):
        return "Missing"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

        return f"{value:.4f}"

    return str(value)


def compare_student_checkpoints(
    student_id: str,
    temporal_datasets: dict[int, pd.DataFrame],
    student_id_columns: dict[int, str | None],
    target_columns: dict[int, str | None],
) -> str:
    """
    Compare one student across all available checkpoints.
    """

    if not temporal_datasets:
        return (
            "No temporal checkpoint datasets are loaded."
        )

    response_lines = [
        f"## Student {student_id} Across Checkpoints",
        "",
    ]

    student_found = False

    preferred_columns = [
        "total_vle_clicks",
        "assessment_count",
        "assessment_score",
        "average_assessment_score",
        "mean_assessment_score",
        "submission_count",
        "num_of_prev_attempts",
        "studied_credits",
        "student_success",
        "success",
        "target",
        "final_result",
        "outcome",
    ]

    for checkpoint_day in sorted(
        temporal_datasets
    ):
        dataset = temporal_datasets[
            checkpoint_day
        ]

        student_id_column = (
            student_id_columns.get(
                checkpoint_day
            )
        )

        target_column = target_columns.get(
            checkpoint_day
        )

        response_lines.extend(
            [
                f"### Day {checkpoint_day}",
                "",
            ]
        )

        if student_id_column is None:
            response_lines.extend(
                [
                    (
                        "Student ID column could not "
                        "be identified."
                    ),
                    "",
                ]
            )

            continue

        try:
            student_records = (
                find_student_at_checkpoint(
                    dataset=dataset,
                    student_id_column=(
                        student_id_column
                    ),
                    student_id=student_id,
                )
            )

        except (ValueError, TypeError):
            response_lines.extend(
                [
                    (
                        "The student ID could not be "
                        "converted to the dataset format."
                    ),
                    "",
                ]
            )

            continue

        if student_records.empty:
            response_lines.extend(
                [
                    (
                        "No record was found for this "
                        "student at this checkpoint."
                    ),
                    "",
                ]
            )

            continue

        student_found = True

        latest_record = (
            student_records.iloc[-1]
        )

        displayed_columns: list[str] = []

        for column_name in preferred_columns:
            if (
                column_name
                in latest_record.index
                and column_name
                != student_id_column
            ):
                formatted_name = (
                    column_name
                    .replace("_", " ")
                    .strip()
                    .title()
                )

                response_lines.append(
                    f"- **{formatted_name}:** "
                    f"{format_value(latest_record[column_name])}"
                )

                displayed_columns.append(
                    column_name
                )

        if (
            target_column is not None
            and target_column
            in latest_record.index
            and target_column
            not in displayed_columns
        ):
            formatted_target_name = (
                target_column
                .replace("_", " ")
                .strip()
                .title()
            )

            response_lines.append(
                f"- **{formatted_target_name}:** "
                f"{format_value(latest_record[target_column])}"
            )

        if not displayed_columns:
            available_columns = [
                column_name
                for column_name
                in latest_record.index
                if column_name
                != student_id_column
            ]

            for column_name in available_columns[
                :8
            ]:
                formatted_name = (
                    column_name
                    .replace("_", " ")
                    .strip()
                    .title()
                )

                response_lines.append(
                    f"- **{formatted_name}:** "
                    f"{format_value(latest_record[column_name])}"
                )

        response_lines.append("")

    if not student_found:
        return (
            f"I could not find student "
            f"**{student_id}** in any loaded "
            "checkpoint dataset."
        )

    response_lines.extend(
        [
            "### Interpretation",
            "",
            (
                "This comparison displays recorded "
                "values from each checkpoint. It does "
                "not yet include model-generated risk "
                "probabilities."
            ),
        ]
    )

    return "\n".join(response_lines)