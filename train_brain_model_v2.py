
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Seed
np.random.seed(42)

# Sample size
num_samples = 1200

# Generate dataset with clinical readings + symptom flags
df = pd.DataFrame({
    'Age': np.random.randint(30, 80, size=num_samples),
    'Sex': np.random.choice(['Male', 'Female'], size=num_samples),
    'BP_Systolic': np.random.randint(110, 200, size=num_samples),
    'BP_Diastolic': np.random.randint(70, 120, size=num_samples),
    'RestingHR': np.random.randint(55, 110, size=num_samples),
    'SpO2': np.random.normal(97, 1.5, size=num_samples).clip(90, 100),
    'FastingBloodSugar': np.random.randint(70, 180, size=num_samples),
    'BMI': np.random.normal(26, 4, size=num_samples).clip(16, 40),
    'StressLevel': np.random.randint(1, 11, size=num_samples),
    'Smokes': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]),
    'BlurredVision': np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15]),
    'FrequentHeadaches': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
    'MobilityDizziness': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),
    'FamilyHistoryBrainEvent': np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])
})

# Label rules (smarter logic)
conditions = (
    (df['BP_Systolic'] >= 170) |
    (df['SpO2'] <= 93) |
    (df['FastingBloodSugar'] >= 160) |
    (df['RestingHR'] >= 100) |
    ((df['BlurredVision'] == 1) & (df['StressLevel'] >= 7)) |
    ((df['MobilityDizziness'] == 1) & (df['BMI'] >= 32))
)

df['RiskLabel'] = np.where(
    conditions, 'EmergencyRisk',
    np.where(
        (df['StressLevel'] >= 6) |
        (df['BMI'] >= 28) |
        (df['BP_Systolic'] >= 145) |
        (df['FastingBloodSugar'] >= 130),
        'Warning',
        'NoRisk'
    )
)

# Train/test split
X = df.drop(columns=['RiskLabel'])
y = df['RiskLabel']

# Define numeric & categorical features
numeric_features = ['Age', 'BP_Systolic', 'BP_Diastolic', 'RestingHR', 'SpO2', 'FastingBloodSugar', 'BMI', 'StressLevel']
categorical_features = ['Sex', 'Smokes', 'BlurredVision', 'FrequentHeadaches', 'MobilityDizziness', 'FamilyHistoryBrainEvent']

# Preprocessing
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X, y)

# Save
joblib.dump(pipeline, "brain_model.pkl")
print("✅ Smart Brain model with clinical data saved as brain_model.pkl")
