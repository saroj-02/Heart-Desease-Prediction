import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ==========================================
# Load Model and Preprocessor
# ==========================================

model = joblib.load(
    "models/logistic_model.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's information below "
    "to predict the likelihood of heart disease."
)

st.info(
    "This application is for educational purposes "
    "and is not a medical diagnosis."
)


# ==========================================
# Patient Information
# ==========================================

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)


sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)


cp = st.selectbox(
    "Chest Pain Type",
    [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ]
)


trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50.0,
    max_value=250.0,
    value=120.0
)


chol = st.number_input(
    "Cholesterol",
    min_value=50.0,
    max_value=600.0,
    value=200.0
)


fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    [True, False]
)


restecg = st.selectbox(
    "Resting ECG",
    [
        "normal",
        "ST-T wave abnormality",
        "left ventricular hypertrophy"
    ]
)


thalch = st.number_input(
    "Maximum Heart Rate",
    min_value=50.0,
    max_value=250.0,
    value=150.0
)


exang = st.selectbox(
    "Exercise Induced Angina",
    [True, False]
)


oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)


slope = st.selectbox(
    "Slope",
    [
        "upsloping",
        "flat",
        "downsloping"
    ]
)


ca = st.number_input(
    "Number of Major Vessels",
    min_value=0.0,
    max_value=4.0,
    value=0.0
)


thal = st.selectbox(
    "Thalassemia",
    [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
)


# ==========================================
# Prediction
# ==========================================

if st.button(
    "Predict Heart Disease",
    type="primary"
):

    # Create dataframe with EXACT
    # column names used during training

    patient_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalch": [thalch],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]
        }
    )


    # ======================================
    # Apply Same Preprocessing
    # ======================================

    patient_processed = preprocessor.transform(
        patient_data
    )


    # ======================================
    # Prediction
    # ======================================

    prediction = model.predict(
        patient_processed
    )[0]


    probability = model.predict_proba(
        patient_processed
    )[0][1]


    # ======================================
    # Display Result
    # ======================================

    st.markdown("---")

    if prediction == 1:

        st.error(
            "⚠️ Higher likelihood of heart disease"
        )

    else:

        st.success(
            "✅ Lower likelihood of heart disease"
        )


    st.metric(
        "Heart Disease Probability",
        f"{probability * 100:.2f}%"
    )


    st.progress(
        float(probability)
    )


    # ======================================
    # Explanation
    # ======================================

    if prediction == 1:

        st.warning(
            "The model predicts a higher likelihood "
            "of heart disease based on the entered "
            "features."
        )

    else:

        st.success(
            "The model predicts a lower likelihood "
            "of heart disease based on the entered "
            "features."
        )