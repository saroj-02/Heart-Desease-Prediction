import joblib
import numpy as np


# Load trained model
model = joblib.load("../models/logistic_model.pkl")

# Load scaler
scaler = joblib.load("../models/scaler.pkl")


def predict_heart_disease(patient_data):

    patient_data = np.array(patient_data).reshape(1, -1)

    # Scale input
    patient_data = scaler.transform(patient_data)

    # Prediction
    prediction = model.predict(patient_data)[0]

    # Probability
    probability = model.predict_proba(patient_data)[0][1]

    if prediction == 1:
        result = "High Risk of Heart Disease"
    else:
        result = "Low Risk of Heart Disease"

    return result, probability


# Example patient
patient = [
    63,     # age
    1,      # sex
    3,      # cp
    145,    # trestbps
    233,    # chol
    1,      # fbs
    0,      # restecg
    150,    # thalach
    0,      # exang
    2.3,    # oldpeak
    0,      # slope
    0,      # ca
    1       # thal
]


result, probability = predict_heart_disease(patient)

print("Prediction:", result)

print(
    "Heart Disease Probability:",
    round(probability * 100, 2),
    "%"
)