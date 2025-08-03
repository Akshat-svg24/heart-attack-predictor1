
import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Heart Attack Risk Predictor")

# Input fields
age = st.number_input("Age", 1, 120)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200)
chol = st.number_input("Cholesterol (chol)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 220)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", 0.0, 10.0)
slope = st.selectbox("Slope of ST segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of major vessels colored (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("High Risk of Heart Attack ⚠️")
    else:
        st.success("Low Risk of Heart Attack ✅")
