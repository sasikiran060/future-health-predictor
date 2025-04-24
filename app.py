# © 2025 Sasi Kiran. All Rights Reserved.
# Future Health Predictor - Predictive Healthcare & Neurological Risk System
# Unauthorized use, reproduction, or distribution is prohibited.
# Future Health Predictor - Unified Streamlit App with All Modules

import streamlit as st
import pandas as pd
import joblib
import os
import random
from fpdf import FPDF
from datetime import datetime

# ========== PDF Export Utility ==========
def export_to_pdf(title, input_dict, insights, score_level):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"{title} Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    for key, val in input_dict.items():
        pdf.multi_cell(200, 8, txt=f"{key}: {val}")

    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt=f"Risk Level: {score_level}", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Why this result:", ln=True)
    pdf.multi_cell(200, 8, txt="\n".join(insights))

    filename = f"{title.lower().replace(' ', '_')}_report.pdf"
    pdf.output(filename)
    st.success(f"📄 PDF exported as {filename}")
    with open(filename, "rb") as f:
        st.download_button("📅 Download Report", f, file_name=filename)

# ========== Model Setup ==========
MODEL_PATHS = {
    "Heart": "timeline_model.pkl",
    "Brain": "brain_model.pkl",
    # Others are simulated
}

# ========== Page Config ==========
st.set_page_config(
    page_title="Future Health Predictor",
    page_icon="🧬",
    layout="centered",
)
st.title("🧠🦠 Future Health Predictor")
st.markdown("Use your vitals to predict and prevent health risks across multiple body systems.")

# ========== Model Loader ==========
def load_model(system):
    model_path = MODEL_PATHS.get(system)
    if not model_path or not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# ========== Sidebar ==========
system_choice = st.sidebar.selectbox(
    "Select Body System",
    ["Heart", "Brain", "Lungs", "Liver", "Kidney", "Diabetes"]
)
model = load_model(system_choice)

# Insert all modules here

# ========== MODULE: BRAIN ==========
if system_choice == "Brain":
    st.header("🧠 Advanced Brain Health & Neurological Risk Analyzer")
    st.markdown("_Combines vital signs, cognitive symptoms, and family history to detect potential neurological risks like stroke, cognitive decline, or neurodegenerative disease._")

    age = st.slider("Age", 18, 100, 45)
    sex = st.radio("Sex", ["Male", "Female"])
    headache = st.radio("Frequent or Severe Headaches?", ["Yes", "No"]) == "Yes"
    blurred_vision = st.radio("Blurred or Double Vision?", ["Yes", "No"]) == "Yes"
    confusion = st.radio("Episodes of Confusion or Memory Lapses?", ["Yes", "No"]) == "Yes"
    numbness = st.radio("Numbness or Weakness in Limbs?", ["Yes", "No"]) == "Yes"
    speech = st.radio("Slurred Speech or Difficulty Speaking?", ["Yes", "No"]) == "Yes"
    sleep_issues = st.radio("Insomnia or Sleep Disturbances?", ["Yes", "No"]) == "Yes"
    family_stroke = st.radio("Family History of Stroke or Brain Disease?", ["Yes", "No"]) == "Yes"
    bp = st.slider("Blood Pressure (mmHg)", 80, 200, 125)
    hr = st.slider("Heart Rate (bpm)", 40, 150, 75)
    spo2 = st.slider("Oxygen Saturation (%)", 85, 100, 96)

    if st.button("🔍 Analyze Brain Health"):
        df_input = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'BP_Systolic': bp,
            'BP_Diastolic': 80,
            'RestingHR': hr,
            'SpO2': spo2,
            'FastingBloodSugar': 100,
            'BMI': 26.0,
            'StressLevel': 5,
            'Smokes': 0,
            'BlurredVision': int(blurred_vision),
            'FrequentHeadaches': int(headache),
            'MobilityDizziness': 0,
            'FamilyHistoryBrainEvent': int(family_stroke)
        }])

        if model:
            prediction = model.predict(df_input)[0]
            if prediction == "NoRisk":
                risk = "Low"
                insights = [
                    "Vitals and clinical markers are within optimal range.",
                    "No neurological red flags detected based on current data."
                ]
            elif prediction == "Warning":
                risk = "Moderate"
                insights = [
                    "Several warning signs of neurological stress detected.",
                    "Suggest early imaging and stress evaluation."
                ]
            else:
                risk = "High"
                insights = [
                    "Serious neurological risk markers present.",
                    "Emergency evaluation required. Risk of stroke or neurovascular incident."
                ]
        else:
            risk = "Unknown"
            insights = ["⚠️ Brain model not found. Please upload 'brain_model.pkl'."]

        st.subheader(f"🧠 Predicted Brain Risk Level: {risk}")
        st.markdown("### 🧠 Why this result:")
        for i in insights:
            st.markdown(f"- {i}")

        export_to_pdf("Brain", df_input.to_dict(orient='records')[0], insights, risk)

# ========== MODULE: HEART ==========
if system_choice == "Heart":
    st.header("🫀 Advanced Cardiovascular Risk Analyzer")
    st.markdown("_Combines clinical tests and subjective symptoms to estimate heart disease risk._")

    age = st.slider("Age", 18, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.radio("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
    bp = st.slider("Resting BP (mmHg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dL)", 100, 400, 210)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"]) == "Yes"
    restecg = st.radio("ECG Results", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
    hr = st.slider("Resting Heart Rate (bpm)", 40, 200, 75)
    exang = st.radio("Exercise-induced Angina?", ["Yes", "No"]) == "Yes"
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
    slope = st.radio("Slope of ST segment", ["Upsloping", "Flat", "Downsloping"])

    if st.button("🔍 Analyze Heart Health"):
        inputs = {
            "Age": age, "Sex": sex, "Chest Pain": cp, "BP": bp, "Cholesterol": chol,
            "FBS > 120": fbs, "RestECG": restecg, "Heart Rate": hr,
            "Exercise Angina": exang, "Oldpeak": oldpeak, "ST Slope": slope
        }

        score = 0
        if cp in ["Typical Angina", "Atypical Angina"]: score += 2
        if bp > 140: score += 1
        if chol > 240: score += 1
        if fbs: score += 1
        if restecg != "Normal": score += 1
        if hr < 60 or hr > 100: score += 1
        if exang: score += 2
        if oldpeak >= 2: score += 1
        if slope == "Flat" or slope == "Downsloping": score += 1

        if score <= 3:
            risk = "Low"
            insights = [
                "Vitals show optimal ranges across all major cardiac indicators.",
                "No chest pain or ECG abnormalities; heart rate is within normal sinus rhythm.",
                "Patient displays healthy metabolic markers—minimal short-term cardiovascular threat."
            ]
        elif score <= 6:
            risk = "Moderate"
            insights = [
                "Mild abnormalities in cholesterol, BP, or ECG indicate early dysfunction.",
                "Symptoms suggest subclinical ischemia or early atherosclerotic changes.",
                "Advise lifestyle modifications, lipid panel, stress echocardiogram for further clarity."
            ]
        else:
            risk = "High"
            insights = [
                "Multiple pathological findings detected including ECG anomalies, stress-induced angina, and metabolic strain.",
                "High probability of myocardial ischemia or evolving coronary artery disease (CAD).",
                "Immediate referral advised for coronary angiography and cardiology evaluation."
            ]

        st.subheader(f"📊 Predicted Heart Risk Level: {risk}")
        st.markdown("### 🔬 Why this result:")
        for i in insights:
            st.markdown(f"- {i}")

        export_to_pdf("Heart", inputs, insights, risk)

# ========== MODULE: LUNGS ==========
if system_choice == "Lungs":
    st.header("🫁 Advanced Lung Health & Respiratory Risk Analyzer")
    st.markdown("_Combines symptoms and vital signs to detect potential respiratory concerns._")

    cough = st.radio("Persistent Cough?", ["Yes", "No"]) == "Yes"
    breathless = st.radio("Shortness of Breath?", ["Yes", "No"]) == "Yes"
    wheeze = st.radio("Wheezing or Noisy Breathing?", ["Yes", "No"]) == "Yes"
    chest_tight = st.radio("Chest Tightness or Pain?", ["Yes", "No"]) == "Yes"
    fatigue = st.radio("Easily Fatigued or Tired?", ["Yes", "No"]) == "Yes"
    smoker = st.radio("Current or Ex-Smoker?", ["Yes", "No"]) == "Yes"
    exposure = st.radio("Exposure to Dust/Chemicals?", ["Yes", "No"]) == "Yes"

    spo2 = st.slider("SpO2 (Oxygen Saturation %)", 80, 100, 96)
    resp_rate = st.slider("Respiratory Rate (breaths/min)", 10, 40, 16)
    hr = st.slider("Heart Rate (bpm)", 40, 140, 75)

    if st.button("🔍 Analyze Lung Health"):
        inputs = {
            "Cough": cough, "Breathless": breathless, "Wheezing": wheeze,
            "Chest Tightness": chest_tight, "Fatigue": fatigue, "Smoker": smoker,
            "Pollutant Exposure": exposure, "SpO2": spo2, "Respiratory Rate": resp_rate, "Heart Rate": hr
        }

        score = sum([cough, breathless, wheeze, chest_tight, fatigue, smoker, exposure])
        if spo2 < 93: score += 2
        if resp_rate > 20: score += 1
        if hr > 100: score += 1

        if score <= 3:
            risk = "Low"
            insights = [
                "Lung function appears stable. Normal oxygen levels and minimal symptoms.",
                "No signs of significant respiratory stress or obstruction."
            ]
        elif score <= 6:
            risk = "Moderate"
            insights = [
                "Some early respiratory indicators noted — e.g. mild cough, exposure, or increased breathing rate.",
                "Could indicate chronic irritation or onset of asthma/bronchitis."
            ]
        else:
            risk = "High"
            insights = [
                "Multiple symptoms suggest obstructive or inflammatory lung disease.",
                "SpO2 below normal and high respiratory rate indicate compromised oxygen exchange.",
                "Recommend pulmonary function tests and chest imaging."
            ]

        st.subheader(f"🫁 Predicted Lung Risk Level: {risk}")
        st.markdown("### 🫁 Why this result:")
        for r in insights:
            st.markdown(f"- {r}")

        export_to_pdf("Lungs", inputs, insights, risk)

# ========== MODULE: LIVER ==========
if system_choice == "Liver":
    st.header("🧬 Liver Function Risk Analyzer")
    st.markdown("_Assesses liver stress based on lab markers and symptoms._")

    age = st.slider("Age", 18, 90, 40)
    sex = st.radio("Sex", ["Male", "Female"])
    fatigue = st.radio("Chronic Fatigue?", ["Yes", "No"]) == "Yes"
    jaundice = st.radio("Yellowing of Eyes/Skin (Jaundice)?", ["Yes", "No"]) == "Yes"
    nausea = st.radio("Nausea or Vomiting?", ["Yes", "No"]) == "Yes"
    swelling = st.radio("Abdominal Swelling?", ["Yes", "No"]) == "Yes"
    alcohol = st.radio("Frequent Alcohol Consumption?", ["Yes", "No"]) == "Yes"

    alt = st.slider("ALT (U/L)", 0, 200, 30)
    ast = st.slider("AST (U/L)", 0, 200, 25)
    bilirubin = st.slider("Bilirubin (mg/dL)", 0.0, 5.0, 0.8)
    albumin = st.slider("Albumin (g/dL)", 2.0, 5.5, 4.0)

    if st.button("🔍 Analyze Liver Health"):
        inputs = {
            "Age": age, "Sex": sex, "Fatigue": fatigue, "Jaundice": jaundice,
            "Nausea": nausea, "Swelling": swelling, "Alcohol": alcohol,
            "ALT": alt, "AST": ast, "Bilirubin": bilirubin, "Albumin": albumin
        }

        score = sum([fatigue, jaundice, nausea, swelling, alcohol])
        if alt > 50 or ast > 50: score += 2
        if bilirubin > 1.2: score += 2
        if albumin < 3.5: score += 1

        if score <= 3:
            risk = "Low"
            insights = [
                "Liver enzyme levels are within safe limits.",
                "No symptoms indicating hepatic dysfunction detected.",
                "Healthy metabolic and protein synthesis profile."
            ]
        elif score <= 6:
            risk = "Moderate"
            insights = [
                "Mild elevation in liver enzymes or early signs of hepatic stress.",
                "May reflect fatty liver, alcohol impact, or early hepatitis."
            ]
        else:
            risk = "High"
            insights = [
                "Multiple elevated markers (ALT/AST/Bilirubin) and symptoms present.",
                "Indicates high risk of liver inflammation or chronic liver disease.",
                "Immediate consultation for ultrasound and LFT panel advised."
            ]

        st.subheader(f"🧬 Predicted Liver Risk Level: {risk}")
        st.markdown("### 🧬 Why this result:")
        for i in insights:
            st.markdown(f"- {i}")

        export_to_pdf("Liver", inputs, insights, risk)

# ========== MODULE: KIDNEY ==========
if system_choice == "Kidney":
    st.header("🩺 Kidney Function & Risk Analyzer")
    st.markdown("_Detects early signs of renal dysfunction based on lab and symptom profiles._")

    age = st.slider("Age", 18, 90, 45)
    sex = st.radio("Sex", ["Male", "Female"])
    urinate = st.radio("Frequent/Painful Urination?", ["Yes", "No"]) == "Yes"
    swelling = st.radio("Swelling in legs/feet?", ["Yes", "No"]) == "Yes"
    blood_urine = st.radio("Blood in Urine?", ["Yes", "No"]) == "Yes"
    fatigue = st.radio("Chronic Fatigue?", ["Yes", "No"]) == "Yes"

    creatinine = st.slider("Creatinine (mg/dL)", 0.5, 5.0, 1.0)
    bun = st.slider("BUN (mg/dL)", 5, 80, 18)
    gfr = st.slider("Estimated GFR (mL/min/1.73m²)", 10, 120, 90)
    albuminuria = st.radio("Albumin in Urine?", ["Yes", "No"]) == "Yes"

    if st.button("🔍 Analyze Kidney Health"):
        inputs = {
            "Age": age, "Sex": sex, "Urination Issues": urinate,
            "Swelling": swelling, "Hematuria": blood_urine, "Fatigue": fatigue,
            "Creatinine": creatinine, "BUN": bun, "GFR": gfr, "Albuminuria": albuminuria
        }

        score = sum([urinate, swelling, blood_urine, fatigue, albuminuria])
        if creatinine > 1.3: score += 2
        if bun > 30: score += 1
        if gfr < 60: score += 2

        if score <= 3:
            risk = "Low"
            insights = [
                "No signs of major renal dysfunction detected.",
                "Normal GFR and creatinine support stable kidney filtration."
            ]
        elif score <= 6:
            risk = "Moderate"
            insights = [
                "Mild elevations in waste markers or urine abnormalities.",
                "Monitor for early nephropathy or glomerular stress."
            ]
        else:
            risk = "High"
            insights = [
                "Multiple risk factors detected: proteinuria, elevated creatinine, reduced GFR.",
                "Suggestive of possible CKD (Chronic Kidney Disease).",
                "Referral to nephrologist and renal imaging recommended."
            ]

        st.subheader(f"🩺 Predicted Kidney Risk Level: {risk}")
        st.markdown("### 🩺 Why this result:")
        for i in insights:
            st.markdown(f"- {i}")

        export_to_pdf("Kidney", inputs, insights, risk)

# ========== MODULE: DIABETES ==========
if system_choice == "Diabetes":
    st.header("🩸 Diabetes Risk & Glycemic Health Analyzer")
    st.markdown("_Combines early symptoms and blood markers to assess prediabetes or diabetes risk._")

    age = st.slider("Age", 18, 90, 40)
    sex = st.radio("Sex", ["Male", "Female"])
    thirsty = st.radio("Increased Thirst?", ["Yes", "No"]) == "Yes"
    frequent_urine = st.radio("Frequent Urination?", ["Yes", "No"]) == "Yes"
    weight_loss = st.radio("Unexplained Weight Loss?", ["Yes", "No"]) == "Yes"
    tired = st.radio("Fatigue or Blurry Vision?", ["Yes", "No"]) == "Yes"
    family_history = st.radio("Family History of Diabetes?", ["Yes", "No"]) == "Yes"

    fbs = st.slider("Fasting Blood Sugar (mg/dL)", 70, 300, 110)
    ppbs = st.slider("Postprandial Blood Sugar (mg/dL)", 100, 400, 160)
    hba1c = st.slider("HbA1c (%)", 4.5, 15.0, 6.0)

    if st.button("🔍 Analyze Diabetes Risk"):
        inputs = {
            "Age": age, "Sex": sex, "Thirst": thirsty, "Frequent Urination": frequent_urine,
            "Weight Loss": weight_loss, "Fatigue": tired, "Family History": family_history,
            "FBS": fbs, "PPBS": ppbs, "HbA1c": hba1c
        }

        score = sum([thirsty, frequent_urine, weight_loss, tired, family_history])
        if fbs > 126: score += 2
        if ppbs > 200: score += 1
        if hba1c >= 6.5: score += 2

        if score <= 3:
            risk = "Low"
            insights = [
                "Blood glucose readings are within normal range.",
                "No persistent diabetic symptoms or family risk detected."
            ]
        elif score <= 6:
            risk = "Moderate"
            insights = [
                "Some sugar markers suggest prediabetic state or early warning.",
                "Combined with symptoms or family history, this indicates moderate risk."
            ]
        else:
            risk = "High"
            insights = [
                "Blood sugar levels and HbA1c are elevated.",
                "Classic diabetic symptoms present — confirmatory tests strongly recommended.",
                "Consult endocrinologist and begin glycemic control strategy."
            ]

        st.subheader(f"🩸 Predicted Diabetes Risk Level: {risk}")
        st.markdown("### 🩸 Why this result:")
        for i in insights:
            st.markdown(f"- {i}")

        export_to_pdf("Diabetes", inputs, insights, risk)

