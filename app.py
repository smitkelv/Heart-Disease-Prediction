import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient information to predict risk of heart disease.")

# -----------------------
# Load and train pipeline (cached)
# -----------------------
@st.cache_resource  # caches the trained pipeline across app sessions
def load_and_train_pipeline():
    # Load CSV (uploaded to repo)
    df = pd.read_excel("heart.xls")
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    # Full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingClassifier())
    ])

    # Train pipeline
    pipeline.fit(X, y)
    return pipeline

# Load pipeline (cached after first run)
model = load_and_train_pipeline()

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
    # Build input dataframe matching training columns
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

    # Predict using the cached pipeline
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Prediction Probability: {probability*100:.2f}%")
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")




