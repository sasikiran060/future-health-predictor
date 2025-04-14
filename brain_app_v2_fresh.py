
import pandas as pd
import joblib
import os

# Load brain model
model_path = "brain_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = joblib.load(model_path)

# Input helper
def get_input(prompt, cast_type=str, allowed=None):
    while True:
        try:
            val = cast_type(input(prompt))
            if allowed and val not in allowed:
                raise ValueError(f"Must be one of: {', '.join(str(a) for a in allowed)}")
            return val
        except ValueError as e:
            print(f"❌ Invalid input: {e}")

print("\n🧠 Smart Brain Health Risk Predictor")
print("Please enter your health data below:\n")

# Medical vitals + symptoms
age = get_input("Age: ", int)
sex = get_input("Sex (Male/Female): ", str, ["Male", "Female"])
bp_sys = get_input("Systolic BP (top number): ", int)
bp_dia = get_input("Diastolic BP (bottom number): ", int)
rest_hr = get_input("Resting Heart Rate (beats/min): ", int)
spo2 = get_input("Oxygen Saturation (SpO2 %): ", float)
fbs = get_input("Fasting Blood Sugar (mg/dL): ", int)
bmi = get_input("BMI (Body Mass Index): ", float)
stress = get_input("Stress Level (1 to 10): ", int)
smokes = get_input("Do you smoke? (0 = No, 1 = Yes): ", int, [0, 1])
blur = get_input("Do you get blurred vision often? (0 = No, 1 = Yes): ", int, [0, 1])
headache = get_input("Do you have frequent headaches? (0 = No, 1 = Yes): ", int, [0, 1])
dizzy = get_input("Do you feel dizzy during walking or stairs? (0 = No, 1 = Yes): ", int, [0, 1])
family_brain = get_input("Family history of brain stroke/death? (0 = No, 1 = Yes): ", int, [0, 1])

# Build input dataframe
input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "BP_Systolic": bp_sys,
    "BP_Diastolic": bp_dia,
    "RestingHR": rest_hr,
    "SpO2": spo2,
    "FastingBloodSugar": fbs,
    "BMI": bmi,
    "StressLevel": stress,
    "Smokes": smokes,
    "BlurredVision": blur,
    "FrequentHeadaches": headache,
    "MobilityDizziness": dizzy,
    "FamilyHistoryBrainEvent": family_brain
}])

# Predict
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
confidence = dict(zip(model.classes_, proba))

# Output
print("\n📋 Risk Prediction:")
if prediction == "NoRisk":
    print("🟢 LOW RISK — Your readings look okay.")
elif prediction == "Warning":
    print("🟠 WARNING — You should monitor closely and improve your lifestyle.")
else:
    print("🔴 EMERGENCY RISK — Seek medical attention urgently!")

print("\nConfidence Levels:")
for k, v in confidence.items():
    print(f"{k}: {v*100:.2f}%")

# Advice
print("\n🧠 Personalized Advice:")
if prediction == "NoRisk":
    print("""
✅ You're in good condition now — great job!

🟢 Tips to Stay Safe:
• Maintain a healthy weight and BMI
• Keep BP and sugar in control with diet & exercise
• Practice stress reduction (breathing, nature, less screen time)
• Sleep well (7–8 hrs)
• Do a full checkup once a year

👨‍⚕️ Ask your doctor: "Can I get a general brain health checkup with ECG, BP, sugar?"
""")
elif prediction == "Warning":
    print("""
🟠 You may be at risk from stress, blood pressure, or sugar levels.

⚠️ What to do:
• Cut down on salt, fried food, sugar
• Walk 30+ mins daily, avoid late nights
• Reduce stress with breathing/yoga/music
• Track BP, HR and sugar weekly

👨‍⚕️ Ask your doctor: "Please check my vitals & brain risk. I want early prevention."
📅 Book a checkup within 2 weeks.
""")
else:
    print("""
🚨 Serious red flags detected — act now!

Symptoms may include: Head pressure, vision issues, unstable walking

🏥 Do this now:
• Go to the nearest hospital or neurologist
• Avoid any stress, intense work, or physical activity
• Eat light, avoid salt completely

👨‍⚕️ Tell your doctor: "I received an AI-based brain risk alert. Please do CT/MRI, ECG, BP & blood work immediately."

⚡ Early action saves lives. Don't wait!
""")

print("\n⚠️ This is an educational tool. Always consult a real doctor for diagnosis.")
