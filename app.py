# © 2025 Sasi Kiran. All Rights Reserved.
# Future Health Predictor - Predictive Healthcare & Neurological Risk System
# Unauthorized use, reproduction, or distribution is prohibited.

# FUTURE HEALTH UI APP - Streamlit Clean Version

import streamlit as st
import pandas as pd
import joblib
import os

# ========== Model Setup ==========
MODEL_PATHS = {
    "Heart": "timeline_model.pkl",
    "Brain": "brain_model.pkl",
    # Add models here when available
    # "Liver": "liver_model.pkl",
    # "Kidney": "kidney_model.pkl",
}

# ========== Page Config ==========
st.set_page_config(
    page_title="Future Health Predictor",
    page_icon="🧬",
    layout="centered",
)
st.title("🧠🫀 Future Health Predictor")
st.markdown("Use your vitals to predict and prevent health risks across multiple body systems.")

# ========== Model Loader ==========
def load_model(system):
    model_path = MODEL_PATHS.get(system)
    if not model_path or not os.path.exists(model_path):
        st.warning(f"Model for {system} is not available. Showing simulated UI.")
        return None
    return joblib.load(model_path)

# ========== Sidebar ==========
system_choice = st.sidebar.selectbox("Select Body System", ["Heart", "Brain", "Lungs (Simulated)", "Diabetes (Simulated)", "Liver (Simulated)", "Kidney (Simulated)"])
model = load_model(system_choice)

# ========== HEART MODULE ==========
if system_choice == "Heart":
    st.header("❤️ Heart Health Check")
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    rbp = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.radio("Fasting Blood Sugar > 120?", ["Yes", "No"])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max HR", 60, 220, 150)
    ex_angina = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("🔍 Predict Heart Health") and model:
        input_df = pd.DataFrame([{
            "Age": age, "Sex": sex[0], "ChestPainType": cp, "RestingBP": rbp, "Cholesterol": chol,
            "FastingBS": 1 if fbs == "Yes" else 0, "RestingECG": ecg, "MaxHR": max_hr,
            "ExerciseAngina": ex_angina[0], "Oldpeak": oldpeak, "ST_Slope": st_slope
        }])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction")
        if prediction == "NoDisease":
            st.success("🟢 Low Risk – Keep up the good health!")
        elif prediction == "LateDiagnosis":
            st.warning("🟠 Moderate Risk – Regular checkups advised.")
        else:
            st.error("🔴 High Risk – Seek medical attention immediately.")

        st.write("**Confidence:**")
        for label, p in zip(model.classes_, proba):
            st.write(f"{label}: {p*100:.2f}%")

# ========== BRAIN MODULE ==========
elif system_choice == "Brain":
    st.header("🧠 Brain Health Check")
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bp_sys = st.number_input("Systolic BP", 90, 200, 130)
    bp_dia = st.number_input("Diastolic BP", 60, 140, 85)
    hr = st.number_input("Resting Heart Rate", 40, 120, 72)
    spo2 = st.slider("Oxygen Saturation (SpO2 %)", 85.0, 100.0, 97.0)
    sugar = st.number_input("Fasting Blood Sugar", 70, 300, 110)
    bmi = st.number_input("BMI", 10.0, 45.0, 25.0)
    stress = st.slider("Stress Level", 1, 10, 5)
    smokes = st.radio("Do you smoke?", ["No", "Yes"]) == "Yes"
    blur = st.radio("Blurred vision?", ["No", "Yes"]) == "Yes"
    headache = st.radio("Frequent headaches?", ["No", "Yes"]) == "Yes"
    dizzy = st.radio("Dizziness on movement?", ["No", "Yes"]) == "Yes"
    family = st.radio("Family history of brain stroke/death?", ["No", "Yes"]) == "Yes"

    if st.button("🔍 Predict Brain Health") and model:
        input_df = pd.DataFrame([{ 
            "Age": age, "Sex": sex, "BP_Systolic": bp_sys, "BP_Diastolic": bp_dia, "RestingHR": hr,
            "SpO2": spo2, "FastingBloodSugar": sugar, "BMI": bmi, "StressLevel": stress,
            "Smokes": int(smokes), "BlurredVision": int(blur), "FrequentHeadaches": int(headache),
            "MobilityDizziness": int(dizzy), "FamilyHistoryBrainEvent": int(family)
        }])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.subheader("Prediction")
        if prediction == "NoRisk":
            st.success("🟢 Low Risk – All vitals look normal!")
        elif prediction == "Warning":
            st.warning("🟠 Warning – Monitor your stress and vitals.")
        else:
            st.error("🔴 Emergency Risk – Seek medical attention!")

        st.write("**Confidence:**")
        for label, p in zip(model.classes_, proba):
            st.write(f"{label}: {p*100:.2f}%")

# ========== LUNGS (Simulated) ==========
elif system_choice == "Lungs (Simulated)":
    st.header("🫁 Lung Health Check (Simulated)")
    st.info("This module is a placeholder. Prediction functionality coming soon.")
    st.text_input("Cough Frequency per Day")
    st.slider("Shortness of Breath (1-10)", 1, 10, 5)
    st.radio("History of Asthma?", ["Yes", "No"])
    st.radio("Exposed to Pollutants Recently?", ["Yes", "No"])
    st.button("Predict Lung Health")

# ========== DIABETES (Simulated) ==========
elif system_choice == "Diabetes (Simulated)":
    st.header("🩸 Diabetes Check (Simulated)")
    st.info("This module is a placeholder. Prediction functionality coming soon.")
    st.number_input("Fasting Glucose Level (mg/dL)", 50, 300, 100)
    st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
    st.radio("Family History of Diabetes?", ["Yes", "No"])
    st.button("Predict Diabetes Risk")

# ========== LIVER (Simulated) ==========
elif system_choice == "Liver (Simulated)":
    st.header("🧪 Liver Health Check (Simulated)")
    st.info("This module is a placeholder. Prediction functionality coming soon.")
    st.number_input("Total Bilirubin", 0.0, 10.0, 1.0)
    st.number_input("Direct Bilirubin", 0.0, 5.0, 0.3)
    st.number_input("Alkaline Phosphotase", 50, 400, 120)
    st.number_input("ALT", 10, 200, 30)
    st.number_input("AST", 10, 200, 30)
    st.number_input("Total Proteins", 4.0, 10.0, 6.5)
    st.number_input("Albumin", 2.0, 6.0, 3.5)
    st.number_input("A/G Ratio", 0.3, 2.5, 1.1)
    st.button("Predict Liver Health")

# ========== KIDNEY (Simulated) ==========
elif system_choice == "Kidney (Simulated)":
    st.header("💧 Kidney Health Check (Simulated)")
    st.info("This module is a placeholder. Prediction functionality coming soon.")
    st.number_input("Blood Pressure", 60, 200, 120)
    st.selectbox("Specific Gravity", ["1.005", "1.010", "1.015", "1.020", "1.025"])
    st.selectbox("Albumin in Urine", ["0", "1", "2", "3", "4", "5"])
    st.selectbox("Sugar in Urine", ["0", "1", "2", "3", "4", "5"])
    st.number_input("Serum Creatinine", 0.5, 10.0, 1.2)
    st.number_input("Hemoglobin", 8.0, 18.0, 13.5)
    st.number_input("Packed Cell Volume", 20, 60, 42)
    st.number_input("WBC Count", 3000, 12000, 7500)
    st.number_input("RBC Count", 2.5, 6.0, 4.5)
    st.radio("Hypertension", ["Yes", "No"])
    st.radio("Diabetes Mellitus", ["Yes", "No"])
    st.button("Predict Kidney Health")

# ========== FOOTER ==========
st.markdown("---")
st.caption("This app provides predictions based on vitals and known patterns. Always consult a doctor for real medical decisions.")
