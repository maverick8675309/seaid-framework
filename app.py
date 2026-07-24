import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SEAID Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("SEAID Student Risk Dashboard")

st.markdown(
    """
    The Student Engagement Analytics and Intervention Dashboard
    uses machine learning to identify students who may benefit
    from proactive academic support.
    """
)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "xgboost_day30.joblib"
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    st.success("Day 30 XGBoost model loaded successfully.")
except FileNotFoundError:
    st.error(
        "The model file could not be found at: "
        f"{MODEL_PATH}"
    )
    st.stop()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# --------------------------------------------------
# Upload student data
# --------------------------------------------------

st.header("Upload Student Data")

uploaded_file = st.file_uploader(
    "Upload a CSV containing Day 30 student features",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        student_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(
            student_data.head(),
            use_container_width=True
        )

        st.write(
            f"Students loaded: {len(student_data):,}"
        )

    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
