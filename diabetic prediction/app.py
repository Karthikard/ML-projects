import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# App Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below to predict the likelihood of diabetes:")

# User Inputs
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
Glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
Age = st.number_input("Age", min_value=0, max_value=120, value=33)

# Prediction Button
if st.button("Predict"):
    # Prepare input
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ The person is likely to have diabetes.")
    else:
        st.success("✅ The person is not likely to have diabetes.")
