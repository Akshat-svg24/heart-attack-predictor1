import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Heart Attack Risk Predictor")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200)
chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [1, 0])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina (exang)", [1, 0])
oldpeak = st.number_input("ST depression (oldpeak)", format="%.1f")
slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])

# Predict
if st.button("Predict"):
    try:
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Attack")
        else:
            st.success("✅ Low Risk of Heart Attack")

    except Exception as e:
        st.error(f"Error: {e}")
