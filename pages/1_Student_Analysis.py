from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.model_loader import load_checkpoint_model
from utils.prediction import predict_student_risk


# -------------------------------------------------------
# Page configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Student Analysis",
    page_icon="🎓",
    layout="wide",
)

st.title("Student Risk Analysis")

st.write(
    "Select a checkpoint and either use a sample student "
    "from the existing checkpoint dataset or upload a "
    "one-row CSV file."
)


# -------------------------------------------------------
# Intervention helper
# -------------------------------------------------------

def get_interventions(
    risk_level: str,
    probability: float,
) -> dict:
    """
    Return recommended intervention steps based on the
    student's predicted risk level.

    These recommendations are rule-based and are not
    generated directly by the machine-learning model.
    """

    if risk_level == "High":
        return {
            "priority": "Immediate",
            "summary": (
                "The student may benefit from prompt, "
                "individualized outreach and coordinated support."
            ),
            "interventions": [
                (
                    "Contact the student within 24–48 hours using "
                    "a supportive, nonjudgmental message."
                ),
                (
                    "Schedule a brief one-on-one meeting to identify "
                    "academic, technological, financial, or personal barriers."
                ),
                (
                    "Review missing assignments, assessment performance, "
                    "attendance, and course engagement with the student."
                ),
                (
                    "Develop a short recovery plan with two or three "
                    "specific and achievable next steps."
                ),
                (
                    "Connect the student with tutoring, advising, counseling, "
                    "financial assistance, accessibility services, or other "
                    "relevant campus resources."
                ),
                (
                    "Establish a follow-up date within one week and monitor "
                    "whether engagement improves."
                ),
            ],
        }

    if risk_level == "Moderate":
        return {
            "priority": "Proactive",
            "summary": (
                "The student may benefit from early outreach before "
                "academic difficulties become more serious."
            ),
            "interventions": [
                (
                    "Send a personalized check-in message recognizing "
                    "the student's current progress."
                ),
                (
                    "Ask whether the student is experiencing barriers "
                    "with course content, technology, time management, "
                    "or outside responsibilities."
                ),
                (
                    "Recommend one relevant support resource, such as "
                    "tutoring, instructor office hours, or academic advising."
                ),
                (
                    "Help the student identify the next important assignment "
                    "or course activity to complete."
                ),
                (
                    "Encourage the student to create a short weekly study "
                    "and assignment plan."
                ),
                (
                    "Review the student's status again at the next checkpoint."
                ),
            ],
        }

    return {
        "priority": "Monitor",
        "summary": (
            "The student is currently predicted to have a lower risk level, "
            "but continued monitoring and encouragement are still appropriate."
        ),
        "interventions": [
            (
                "Provide positive reinforcement and acknowledge "
                "the student's continued engagement."
            ),
            (
                "Remind the student about upcoming assignments, assessments, "
                "and available support services."
            ),
            (
                "Encourage continued participation in course activities."
            ),
            (
                "Monitor for meaningful changes in attendance, assignment "
                "completion, assessment performance, or online engagement."
            ),
            (
                "Reassess the student at the next checkpoint."
            ),
        ],
    }


# -------------------------------------------------------
# Checkpoint selection
# -------------------------------------------------------

checkpoint_day = st.selectbox(
    "Select checkpoint",
    options=[7, 14, 21, 30],
    format_func=lambda day: f"Day {day}",
)


# -------------------------------------------------------
# Choose student data source
# -------------------------------------------------------

input_method = st.radio(
    "Choose student data source",
    options=[
        "Use a sample student",
        "Upload a student CSV",
    ],
    horizontal=True,
)

student_data = None


# -------------------------------------------------------
# Use sample student
# -------------------------------------------------------

if input_method == "Use a sample student":
    dataset_path = (
        Path("data")
        / "processed"
        / f"early_warning_day{checkpoint_day}_dataset.csv"
    )

    if not dataset_path.exists():
        st.error(
            "The selected checkpoint dataset was not found."
        )

        st.code(str(dataset_path))

    else:
        try:
            full_dataset = pd.read_csv(
                dataset_path
            )

            if full_dataset.empty:
                st.warning(
                    "The selected checkpoint dataset is empty."
                )

            else:
                sample_row_number = st.number_input(
                    "Select sample row",
                    min_value=1,
                    max_value=len(full_dataset),
                    value=1,
                    step=1,
                )

                student_data = full_dataset.iloc[
                    [int(sample_row_number) - 1]
                ].copy()

                st.success(
                    f"Loaded row {sample_row_number} "
                    f"from the Day {checkpoint_day} dataset."
                )

        except Exception as error:
            st.error(
                "The sample dataset could not be loaded."
            )

            st.exception(error)


# -------------------------------------------------------
# Upload student CSV
# -------------------------------------------------------

else:
    uploaded_file = st.file_uploader(
        "Upload student CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            student_data = pd.read_csv(
                uploaded_file
            )

        except Exception as error:
            st.error(
                "The uploaded CSV could not be read."
            )

            st.exception(error)


# -------------------------------------------------------
# Display student record
# -------------------------------------------------------

if student_data is not None:
    st.subheader("Student Record")

    st.dataframe(
        student_data,
        use_container_width=True,
        hide_index=True,
    )

    if len(student_data) != 1:
        st.warning(
            "The student data must contain exactly one row."
        )

    else:
        if st.button(
            "Generate Risk Prediction",
            type="primary",
        ):
            try:
                # ---------------------------------------
                # Load model and preprocessing pipeline
                # ---------------------------------------

                preprocessing, model = (
                    load_checkpoint_model(
                        checkpoint_day
                    )
                )

                expected_columns = list(
                    preprocessing.feature_names_in_
                )

                missing_columns = [
                    column
                    for column in expected_columns
                    if column not in student_data.columns
                ]

                extra_columns = [
                    column
                    for column in student_data.columns
                    if column not in expected_columns
                ]

                # ---------------------------------------
                # Validate student data
                # ---------------------------------------

                if missing_columns:
                    st.error(
                        "The student record is missing "
                        "required model columns."
                    )

                    with st.expander(
                        "View missing columns"
                    ):
                        st.write(
                            missing_columns
                        )

                else:
                    if extra_columns:
                        st.info(
                            f"{len(extra_columns)} extra "
                            "column(s) were ignored."
                        )

                        with st.expander(
                            "View ignored columns"
                        ):
                            st.write(
                                extra_columns
                            )

                    model_input = student_data[
                        expected_columns
                    ].copy()

                    # ---------------------------------------
                    # Generate prediction
                    # ---------------------------------------

                    result = predict_student_risk(
                        student_data=model_input,
                        preprocessing=preprocessing,
                        model=model,
                    )

                    prediction_label = (
                        "At Risk"
                        if result["predicted_class"] == 1
                        else "Not At Risk"
                    )

                    # ---------------------------------------
                    # Display prediction
                    # ---------------------------------------

                    st.subheader("Prediction Result")

                    col1, col2, col3 = st.columns(
                        3
                    )

                    col1.metric(
                        "Risk Probability",
                        f"{result['probability']:.1%}",
                    )

                    col2.metric(
                        "Risk Level",
                        result["risk_level"],
                    )

                    col3.metric(
                        "Prediction",
                        prediction_label,
                    )

                    if result["predicted_class"] == 1:
                        st.warning(
                            "This student is currently classified "
                            "as at risk based on the selected "
                            "checkpoint model."
                        )
                    else:
                        st.success(
                            "This student is currently classified "
                            "as not at risk based on the selected "
                            "checkpoint model."
                        )

                    # ---------------------------------------
                    # Intervention recommendations
                    # ---------------------------------------

                    intervention_plan = get_interventions(
                        risk_level=result["risk_level"],
                        probability=result["probability"],
                    )

                    st.subheader(
                        "Recommended Interventions"
                    )

                    priority_col, checkpoint_col = st.columns(
                        2
                    )

                    priority_col.metric(
                        "Intervention Priority",
                        intervention_plan["priority"],
                    )

                    checkpoint_col.metric(
                        "Assessment Checkpoint",
                        f"Day {checkpoint_day}",
                    )

                    st.write(
                        intervention_plan["summary"]
                    )

                    for intervention in intervention_plan[
                        "interventions"
                    ]:
                        st.markdown(
                            f"- {intervention}"
                        )

                    # ---------------------------------------
                    # Suggested outreach message
                    # ---------------------------------------

                    with st.expander(
                        "View suggested student outreach message"
                    ):
                        if result["risk_level"] == "High":
                            st.write(
                                "Hi, I wanted to check in because I noticed "
                                "that you may be experiencing some difficulty "
                                "in the course. I would like to help you identify "
                                "any barriers and create a manageable plan for "
                                "moving forward. Please let me know a good time "
                                "for us to connect."
                            )

                        elif result["risk_level"] == "Moderate":
                            st.write(
                                "Hi, I wanted to check in and see how things "
                                "are going in the course. I noticed that you "
                                "may benefit from some additional support. "
                                "Please let me know if you are experiencing "
                                "any challenges or if I can help you connect "
                                "with course or campus resources."
                            )

                        else:
                            st.write(
                                "Hi, I wanted to recognize your continued "
                                "progress in the course. Please remember that "
                                "support is available if you have questions "
                                "or encounter any challenges. Keep up the "
                                "good work."
                            )

                    # ---------------------------------------
                    # Responsible-use note
                    # ---------------------------------------

                    st.info(
                        "These interventions are rule-based recommendations "
                        "intended to support professional judgment. The model "
                        "does not identify the cause of a student's risk and "
                        "should not be used as the sole basis for academic or "
                        "administrative decisions."
                    )

                    st.success(
                        "Prediction and intervention plan "
                        "completed successfully."
                    )

            except Exception as error:
                st.error(
                    "The prediction could not be generated."
                )

                st.exception(error)