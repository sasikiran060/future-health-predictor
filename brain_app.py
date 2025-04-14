
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

print("\n🧠 Brain Health Risk Predictor")
print("Please answer the following questions to assess your risk:\n")

# Gather input
age = get_input("Age: ", int)
sex = get_input("Sex (Male/Female): ", str, ["Male", "Female"])
bp_sys = get_input("Systolic BP (top number, e.g. 130): ", int)
bp_dia = get_input("Diastolic BP (bottom number, e.g. 85): ", int)
has_htn = get_input("Have you been diagnosed with high blood pressure? (0 = No, 1 = Yes): ", int, [0, 1])
stress = get_input("On a scale of 1–10, how stressed are you generally?: ", int)
smokes = get_input("Do you smoke? (0 = No, 1 = Yes): ", int, [0, 1])
blur = get_input("Do you experience blurred vision often? (0 = No, 1 = Yes): ", int, [0, 1])
headache = get_input("Do you get frequent headaches? (0 = No, 1 = Yes): ", int, [0, 1])
dizzy = get_input("Do you feel dizzy while walking or climbing stairs? (0 = No, 1 = Yes): ", int, [0, 1])
family_brain = get_input("Family history of brain stroke or sudden death? (0 = No, 1 = Yes): ", int, [0, 1])

# Create DataFrame
input_df = pd.DataFrame([{
    "Age": age,
    "Sex": sex,
    "BP_Systolic": bp_sys,
    "BP_Diastolic": bp_dia,
    "HasHypertension": has_htn,
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

# Show result
print("\nPrediction Result:")
if prediction == "NoRisk":
    print("🟢 LOW RISK — Your brain health looks stable.")
elif prediction == "Warning":
    print("🟠 MODERATE RISK — Some signs are concerning. Start preventive care.")
else:
    print("🔴 HIGH RISK — Seek medical help immediately.")

print("\nConfidence Levels:")
for k, v in confidence.items():
    print(f"{k}: {v*100:.2f}%")

# Show personalized advice
print("\n📋 Advice:")

if prediction == "NoRisk":
    print("""
✅ Your current brain health indicators seem fine.

What you can keep doing:
• Maintain a low-salt, low-oil diet
• Exercise lightly every day
• Sleep 7–8 hours
• Manage screen time and stress
• Get a full health check once a year

🧠 Ask your doctor: "Can I get a BP, eye, and general brain wellness check?"
""")
elif prediction == "Warning":
    print("""
⚠️ Some early warning signs of brain stress or BP-related damage.

What you should do:
• Reduce mental stress (breathing, yoga, nature walks)
• Limit coffee, junk food, late nights
• Track BP daily or weekly if possible
• Avoid overworking, especially in the sun or long hours

🧠 Ask your doctor: "Can I get a brain health screen — BP, stress, cholesterol, ECG?"

📅 Visit your doctor within the next few weeks for early management.
""")
else:
    print("""
🚨 Serious risk detected — do not delay.

Possible symptoms: pressure in head, blurred vision, confusion, unsteady walking.

What to do immediately:
• Go to a nearby hospital or neurology clinic
• Stop heavy lifting, long-distance walks, or solo travel
• Avoid loud noise, stress, or emotional triggers

🧠 Ask your doctor: "I’m getting a brain warning. Please check BP, brain scan (CT or MRI), and heart condition."

🏥 Emergency care is better than regret. Act now.
""")

print("\n⚠️ This is an educational tool. Consult a real doctor for diagnosis.")
