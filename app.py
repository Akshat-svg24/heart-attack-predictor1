import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Heart Attack Risk Predictor")
st.write("Predict whether you're at **Low**, **Moderate**, or **High** risk of heart attack.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200)
chol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", min_value=60, max_value=250)
exang = st.selectbox("Exercise Induced Angina", [1, 0])
oldpeak = st.number_input("ST depression", format="%.1f")
slope = st.selectbox("Slope", [0, 1, 2])

# Predict
if st.button("Predict"):
    try:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict_proba(input_scaled)[0][1]

        if prediction > 0.75:
            st.error("🔴 High Risk of Heart Attack")
        elif prediction > 0.4:
            st.warning("🟠 Moderate Risk of Heart Attack")
        else:
            st.success("🟢 Low Risk of Heart Attack")

        st.write(f"**Model Confidence**: {round(prediction * 100, 2)}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
