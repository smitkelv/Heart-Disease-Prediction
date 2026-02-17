import streamlit as st
import pandas as pd
import joblib  # for loading your saved model

st.title("Heart Disease Prediction")
st.write("Enter patient information to predict risk of heart disease.")

# -----------------------
# Load the trained model
# -----------------------
try:
    model = joblib.load("heart_model.pkl")  # your saved GradientBoostingClassifier
except FileNotFoundError:
    st.error("Model file 'heart_model.pkl' not found. Upload it to the repo root.")
    st.stop()

# -----------------------
# User Inputs
# -----------------------
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

# -----------------------
# Predict Button
# -----------------------
if st.button("Predict"):

    # Build DataFrame for the model
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

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Prediction Probability: {probability*100:.2f}%")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")






