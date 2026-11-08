
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")
st.subheader("Machine Learning Interactive Prediction System")

st.write(
    "Enter sample measurements below to see the output "
    "of our trained machine learning model."
)

st.info(
    "Educational demonstration only. This application "
    "does not diagnose heart disease or provide medical advice."
)

# Load model once and cache it
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Model file not found.")

    return joblib.load(model_path)

# Load model safely
try:
    saved_data = load_model()
    model = saved_data["model"]
    features = saved_data["features"]

except FileNotFoundError:
    st.error(
        "The trained model is missing. "
        "Please run train_model.py first."
    )
    st.stop()

except Exception:
    st.error(
        "Unable to load the model. "
        "Please check model.joblib and package versions."
    )
    st.stop()

st.divider()

st.header("Enter Sample Measurements")

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 90, 45)

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical angina",
            2: "Atypical angina",
            3: "Non-anginal pain",
            4: "Asymptomatic"
        }[x]
    )

    trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        80, 220, 120
    )

    chol = st.slider(
        "Serum Cholesterol (mg/dl)",
        100, 600, 200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )

with col2:
    thalach = st.slider(
        "Maximum Heart Rate Achieved",
        60, 220, 150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        0.0, 6.0, 1.0, step=0.1
    )

    slope = st.selectbox(
        "ST Segment Slope",
        [1, 2, 3]
    )

    ca = st.selectbox(
        "Major Vessels (0-3)",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [3, 6, 7],
        format_func=lambda x: {
            3: "Normal",
            6: "Fixed defect",
            7: "Reversible defect"
        }[x]
    )

st.divider()

# Prediction
if st.button("Predict", type="primary", use_container_width=True):

    try:
        input_data = pd.DataFrame(
            [[
                age, sex, cp, trestbps, chol,
                fbs, restecg, thalach, exang,
                oldpeak, slope, ca, thal
            ]],
            columns=features
        )

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        # Find probability of predicted class
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100

        st.header("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model predicts the positive "
                "heart disease class."
            )
        else:
            st.success(
                "The model predicts the negative "
                "heart disease class."
            )

        st.metric(
            "Model probability for predicted class",
            f"{confidence:.2f}%"
        )

        st.caption(
            "This percentage is the model's estimated "
            "probability for its predicted class. It is "
            "not a person's clinical probability or diagnosis."
        )

        st.progress(float(confidence / 100))

        st.write("Class probabilities")

        probability_df = pd.DataFrame({
            "Class": ["No disease (0)", "Disease (1)"],
            "Model probability": [
                probabilities[list(model.classes_).index(0)],
                probabilities[list(model.classes_).index(1)]
            ]
        })

        st.bar_chart(
            probability_df.set_index("Class")
        )

    except Exception:
        st.error(
            "Prediction could not be completed. "
            "Please check your inputs and model file."
        )

st.divider()

st.caption(
    "Built with Python, Scikit-learn, Joblib and Streamlit. "
    "Dataset: UCI Cleveland Heart Disease dataset."
)
