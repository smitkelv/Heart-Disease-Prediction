import streamlit as st
import pandas as pd
import joblib

st.title("Heart Disease Prediction")
st.write("Enter patient information to predict risk of heart disease.")

# -----------------------
# Load trained model
# -----------------------
try:
    model = joblib.load("heart_model.pkl")
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
# Manual Encoding
# -----------------------
def encode_input(df):
    # Sex: M=1, F=0
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})

    # ChestPainType one-hot encoding
    for val in ["ATA", "NAP", "ASY", "TA"]:
        df[f"ChestPainType_{val}"] = (df["ChestPainType"] == val).astype(int)
    df = df.drop("ChestPainType", axis=1)

    # RestingECG one-hot
    for val in ["Normal", "ST", "LVH"]:
        df[f"RestingECG_{val}"] = (df["RestingECG"] == val).astype(int)
    df = df.drop("RestingECG", axis=1)

    # ExerciseAngina: Y=1, N=0
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

    # ST_Slope one-hot
    for val in ["Up", "Flat", "Down"]:
        df[f"ST_Slope_{val}"] = (df["ST_Slope"] == val).astype(int)
    df = df.drop("ST_Slope", axis=1)

    return df

# -----------------------
# Predict Button
# -----------------------
if st.button("Predict"):

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

    input_encoded = encode_input(input_data)

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.write(f"Prediction Probability: {probability*100:.2f}%")
    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")







