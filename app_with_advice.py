
import pandas as pd
import joblib
import os

# Load the trained model
model_path = "timeline_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = joblib.load(model_path)

# Collect input from user
print("\n🩺 Future Health Risk Predictor")
print("Please enter the following patient data:\n")

def get_input(prompt, cast_type=str, allowed=None):
    while True:
        try:
            val = cast_type(input(prompt))
            if allowed and val not in allowed:
                raise ValueError(f"Must be one of: {', '.join(str(a) for a in allowed)}")
            return val
        except ValueError as e:
            print(f"❌ Invalid input: {e}")

# Get user inputs
age = get_input("Age (years): ", int)
sex = get_input("Sex (Male/Female): ", str, ["Male", "Female"])
chol = get_input("Cholesterol (mg/dL): ", int)
max_hr = get_input("Max Heart Rate: ", int)
st_slope = get_input("ST Slope (Up / Flat / Down): ", str, ["Up", "Flat", "Down"])
fasting_bs = get_input("Fasting Blood Sugar > 120 mg/dL? (0 = No, 1 = Yes): ", int, [0, 1])
resting_bp = get_input("Resting Blood Pressure: ", int)
exercise_angina = get_input("Exercise-induced Angina? (Yes/No): ", str, ["Yes", "No"])

# Prepare input for prediction
input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "Cholesterol": chol,
    "MaxHR": max_hr,
    "ST_Slope": st_slope,
    "FastingBS": fasting_bs,
    "RestingBP": resting_bp,
    "ExerciseAngina": exercise_angina
}])

# Predict
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
risk_confidence = dict(zip(model.classes_, proba))

# Output
print("\nPrediction Summary:")
print(input_df.to_string(index=False))

print("\nPrediction Result:")
if prediction == "NoDisease":
    print("🟢 Low Risk — You’re doing well, keep it up!")
elif prediction == "LateDiagnosis":
    print("🟠 Possible future risk — stay alert and monitor health regularly.")
else:
    print("🔴 High Risk — immediate medical checkup is recommended.")

print("\nConfidence Levels:")
for label, prob in risk_confidence.items():
    print(f"{label}: {prob*100:.2f}%")

# Custom advice block
print("\n🧠 Personalized Health Advice:")
if prediction == "NoDisease":
    print("""
✅ You’re doing great! Keep maintaining your healthy lifestyle.

💡 Prevention Tips:
• 🥗 Eat a balanced diet (fruits, vegetables, low oil/salt)
• 🚶‍♂️ Walk daily 30 mins or do light cardio
• 🚭 Avoid smoking & heavy alcohol
• 😴 Sleep 7–8 hours
• 🩺 Get yearly health checkups
👨‍⚕️ Ask your doctor: "Can I get a routine heart checkup?"
""")
elif prediction == "LateDiagnosis":
    print("""
🟠 You may be developing early warning signs.

💡 What to do:
• 🧂 Cut salt & sugar
• 🚶 Walk 30+ mins/day
• 🍽️ Reduce fried, red meat, packaged food
• 🧘‍♂️ Try deep breathing or light meditation
• 📈 Track BP weekly if possible
👨‍⚕️ Tell your doctor: "I’d like to check my heart, including cholesterol, ECG & BP."
🗓️ Recommended visit: Within 1–3 months
""")
elif prediction == "SuddenDeath":
    print("""
🔴 Very high risk — urgent care may be needed.

⚠️ What to watch:
• Chest pain, fast heartbeat, shortness of breath
• Pain in left arm, jaw, back

🏥 What to ask for:
• ECG (heart test)
• Cholesterol & BP test
• Stress test or Echo (if advised)

🛑 Avoid stress, smoking, heavy lifting or late nights
👨‍⚕️ Tell your doctor: "I’m getting a digital heart warning and need a full cardiac checkup."
""")

print("\n⚠️ This is a simulated prediction. For real conditions, consult your doctor.")
