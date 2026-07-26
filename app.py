import streamlit as st

st.set_page_config(
    page_title="SEAID Dashboard",
    page_icon="🎓",
    layout="wide",
)

st.title("SEAID")

st.subheader(
    "Student Explainable Artificial Intelligence and "
    "Deep Learning Framework for Educational Decision Support"
)

st.write(
    "Ask SEAID a question about the framework, models, "
    "temporal checkpoints, or student-support indicators."
)

# Store the conversation between Streamlit reruns.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Welcome to SEAID. What would you like to know?"
            ),
        }
    ]


def answer_query(query: str) -> str:
    """
    Return a temporary rule-based answer.

    This function will later be connected to the SEAID dataset,
    trained models, and explainability outputs.
    """

    normalized_query = query.lower().strip()

    if "what is seaid" in normalized_query:
        return (
            "SEAID is the Student Explainable Artificial Intelligence "
            "and Deep Learning Framework for Educational Decision Support. "
            "It combines temporal prediction, explainable AI, deep learning, "
            "and human-centered educational decision support."
        )

    if "model" in normalized_query:
        return (
            "SEAID currently evaluates Logistic Regression, Random Forest, "
            "XGBoost, and neural-network models."
        )

    if "checkpoint" in normalized_query or "day 7" in normalized_query:
        return (
            "SEAID uses temporal checkpoints to evaluate student indicators "
            "at defined points in a course, such as Day 7, Day 14, and Day 21."
        )

    if "priority review" in normalized_query:
        return (
            "Priority Review means the current record contains indicators "
            "that may warrant timely human review. It is not a final judgment "
            "about the student or an automatic intervention decision."
        )

    if "moderate review" in normalized_query:
        return (
            "Moderate Review means the student record contains mixed "
            "indicators. An educator may want to review recent engagement, "
            "assessment, and participation patterns."
        )

    if "responsible" in normalized_query or "ethics" in normalized_query:
        return (
            "SEAID is designed as a human-in-the-loop decision-support "
            "framework. Its predictions should support professional judgment, "
            "not automatically penalize, exclude, or label students."
        )

    return (
        "I do not have that information connected yet. "
        "The next stage will connect this query interface to the SEAID "
        "dataset, trained models, and explainability results."
    )


# Display previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept a new question.
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

    response = answer_query(user_query)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)
