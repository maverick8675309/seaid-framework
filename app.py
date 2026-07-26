from __future__ import annotations

import streamlit as st

from src.data_loader import (
    find_processed_csv_files,
    find_student_id_column,
    find_target_column,
    load_processed_dataset,
    load_temporal_datasets,
    summarize_dataset,
)
from src.query_engine import answer_query
from src.temporal_analysis import (
    build_checkpoint_map,
)


st.set_page_config(
    page_title="SEAID Dashboard",
    page_icon="🎓",
    layout="wide",
)


st.title("SEAID")

st.subheader(
    "Student Explainable Artificial Intelligence "
    "and Deep Learning Framework for "
    "Educational Decision Support"
)


# -------------------------------------------------------
# Find checkpoint files
# -------------------------------------------------------

csv_files = find_processed_csv_files()

checkpoint_map = build_checkpoint_map(
    csv_files
)

dataset = None
dataset_summary = None
student_id_column = None
target_column = None

temporal_datasets = {}
student_id_columns = {}
target_columns = {}


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:
    st.header("Temporal Checkpoint")

    if not checkpoint_map:
        st.warning(
            "No temporal checkpoint CSV files "
            "were found."
        )

        st.caption(
            "Place Day 7, Day 14, Day 21, "
            "and Day 30 CSV files in "
            "`data/processed/`."
        )

    else:
        checkpoint_days = list(
            checkpoint_map.keys()
        )

        selected_checkpoint = st.selectbox(
            "Select checkpoint",
            options=checkpoint_days,
            format_func=lambda day: (
                f"Day {day}"
            ),
        )

        selected_file = checkpoint_map[
            selected_checkpoint
        ]

        try:
            dataset = load_processed_dataset(
                str(selected_file)
            )

            dataset_summary = (
                summarize_dataset(
                    dataset
                )
            )

            student_id_column = (
                find_student_id_column(
                    dataset
                )
            )

            target_column = (
                find_target_column(
                    dataset
                )
            )

            path_mapping = {
                day: str(path)
                for day, path
                in checkpoint_map.items()
            }

            temporal_datasets = (
                load_temporal_datasets(
                    path_mapping
                )
            )

            student_id_columns = {
                day: (
                    find_student_id_column(
                        checkpoint_dataset
                    )
                )
                for day, checkpoint_dataset
                in temporal_datasets.items()
            }

            target_columns = {
                day: find_target_column(
                    checkpoint_dataset
                )
                for day, checkpoint_dataset
                in temporal_datasets.items()
            }

            st.success(
                f"Loaded Day "
                f"{selected_checkpoint}"
            )

            st.caption(
                selected_file.name
            )

            st.metric(
                "Rows",
                f"{dataset_summary['rows']:,}",
            )

            st.metric(
                "Columns",
                f"{dataset_summary['columns']:,}",
            )

        except Exception as error:
            st.error(
                "The checkpoint data could "
                f"not be loaded: {error}"
            )

            dataset = None

    st.divider()

    st.header("Example Questions")

    st.markdown(
        """
        - What is SEAID?
        - Which model performs best?
        - How many students are loaded?
        - What is the target column?
        - Analyze student 1003.
        - Compare student 1003 across checkpoints.
        - Show the target distribution.
        """
    )

    if st.button(
        "Clear Conversation"
    ):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Conversation cleared. "
                    "What would you like "
                    "to know?"
                ),
            }
        ]

        st.rerun()


# -------------------------------------------------------
# Checkpoint summary
# -------------------------------------------------------

if checkpoint_map:
    checkpoint_columns = st.columns(
        len(checkpoint_map)
    )

    for column, checkpoint_day in zip(
        checkpoint_columns,
        checkpoint_map,
    ):
        checkpoint_dataset = (
            temporal_datasets.get(
                checkpoint_day
            )
        )

        if checkpoint_dataset is not None:
            with column:
                st.metric(
                    f"Day {checkpoint_day}",
                    (
                        f"{len(checkpoint_dataset):,} "
                        "records"
                    ),
                )


# -------------------------------------------------------
# Dataset preview
# -------------------------------------------------------

if dataset is not None:
    with st.expander(
        "Preview Selected Checkpoint"
    ):
        st.dataframe(
            dataset.head(20),
            use_container_width=True,
            hide_index=True,
        )


# -------------------------------------------------------
# Conversation state
# -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to SEAID. "
                "You can query the selected "
                "checkpoint or compare a student "
                "across all checkpoints."
            ),
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# -------------------------------------------------------
# Chat
# -------------------------------------------------------

user_query = st.chat_input(
    "Ask SEAID a question"
)


if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    response = answer_query(
        query=user_query,
        dataset=dataset,
        dataset_summary=(
            dataset_summary
        ),
        student_id_column=(
            student_id_column
        ),
        target_column=target_column,
        temporal_datasets=(
            temporal_datasets
        ),
        student_id_columns=(
            student_id_columns
        ),
        target_columns=(
            target_columns
        ),
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    with st.chat_message(
        "assistant"
    ):
        st.markdown(response)