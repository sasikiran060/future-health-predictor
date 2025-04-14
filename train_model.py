# © 2025 Sasi Kiran. All Rights Reserved.
# Future Health Predictor - Predictive Healthcare & Neurological Risk System
# Unauthorized use, reproduction, or distribution is prohibited.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Generate synthetic patient dataset with clinical features
np.random.seed(42)
num_patients = 1000
records_per_patient = 3
start_date = datetime(2018, 1, 1)
data = []
outcomes = ['NoDisease', 'LateDiagnosis', 'SuddenDeath']
weights = [0.75, 0.15, 0.10]

for pid in range(1, num_patients + 1):
    age = np.random.randint(30, 70)
    sex = np.random.choice(['Male', 'Female'])
    fasting_bs = np.random.choice([0, 1], p=[0.8, 0.2])
    exercise_angina = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
    outcome = np.random.choice(outcomes, p=weights)

    base_chol = np.random.randint(160, 240)
    base_hr = np.random.randint(120, 180)
    base_st = np.random.choice(['Up', 'Flat', 'Down'], p=[0.6, 0.3, 0.1])
    base_bp = np.random.randint(100, 160)

    for visit in range(records_per_patient):
        checkup_date = start_date + timedelta(days=365 * visit + np.random.randint(-30, 30))
        cholesterol = base_chol + np.random.normal(0, 12) + visit * (5 if outcome != 'NoDisease' else 0)
        max_hr = base_hr - visit * 2 + np.random.normal(0, 5)
        st_slope = np.random.choice(['Up', 'Flat', 'Down'], p=[0.5, 0.35, 0.15]) if visit > 0 else base_st
        resting_bp = base_bp + np.random.normal(0, 5) + (3 if outcome == 'SuddenDeath' else 0)

        data.append({
            'PatientID': pid,
            'CheckupDate': checkup_date.strftime("%Y-%m-%d"),
            'Age': age + visit,
            'Sex': sex,
            'Cholesterol': round(cholesterol),
            'MaxHR': round(max_hr),
            'ST_Slope': st_slope,
            'FastingBS': fasting_bs,
            'RestingBP': round(resting_bp),
            'ExerciseAngina': exercise_angina,
            'FinalOutcome': outcome
        })

df = pd.DataFrame(data)

# Use only first record for each patient
first_records = df.groupby('PatientID').first().reset_index()
X = first_records.drop(columns=['PatientID', 'CheckupDate', 'FinalOutcome'])
y = first_records['FinalOutcome']

# Define numeric and categorical features
numeric_features = ['Age', 'Cholesterol', 'MaxHR', 'RestingBP']
categorical_features = ['Sex', 'ST_Slope', 'FastingBS', 'ExerciseAngina']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "timeline_model.pkl")
print("✅ Model trained and saved as timeline_model.pkl")
