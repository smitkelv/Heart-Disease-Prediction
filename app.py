import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load trained pipeline
# ---------------------------
model = joblib.load("heart_pipeline2.pkl")

st.title("Heart Disease Risk Predictor")
st.write("Enter patient information to predict heart disease risk:")

# ---------------------------
# User Inputs
# ---------------------------
age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", ["M", "F"])
chestpain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
restingbp = st.number_input("Resting BP", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
fastingbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restingecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
maxhr = st.number_input("Max HR", 60, 220, 150)
exerciseangina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
stslope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict"):

    # Create input dataframe matching original dataset
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chestpain,
        "RestingBP": restingbp,
        "Cholesterol": cholesterol,
        "FastingBS": fastingbs,
        "RestingECG": restingecg,
        "MaxHR": maxhr,
        "ExerciseAngina": exerciseangina,
        "Oldpeak": oldpeak,
        "ST_Slope": stslope
    }])

    # Use full pipeline to predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of heart disease

    # Display results
    st.write(f"Prediction Probability: {probability*100:.2f}%")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
