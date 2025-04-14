
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Set seed
np.random.seed(42)

# Generate synthetic data
num_samples = 1000
df = pd.DataFrame({
    'Age': np.random.randint(30, 80, size=num_samples),
    'Sex': np.random.choice(['Male', 'Female'], size=num_samples),
    'BP_Systolic': np.random.randint(110, 200, size=num_samples),
    'BP_Diastolic': np.random.randint(70, 120, size=num_samples),
    'HasHypertension': np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6]),
    'StressLevel': np.random.randint(1, 11, size=num_samples),
    'Smokes': np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3]),
    'BlurredVision': np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15]),
    'FrequentHeadaches': np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]),
    'MobilityDizziness': np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]),
    'FamilyHistoryBrainEvent': np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])
})

# Define labels based on logic
conditions = (
    (df['BP_Systolic'] > 160) & 
    (df['StressLevel'] > 7) & 
    ((df['BlurredVision'] == 1) | (df['FrequentHeadaches'] == 1) | (df['MobilityDizziness'] == 1))
)

df['RiskLabel'] = np.where(conditions, 'EmergencyRisk', 
                           np.where((df['HasHypertension'] == 1) | (df['StressLevel'] >= 6), 'Warning', 'NoRisk'))

X = df.drop(columns=['RiskLabel'])
y = df['RiskLabel']

# Define feature types
numeric_features = ['Age', 'BP_Systolic', 'BP_Diastolic', 'StressLevel']
categorical_features = ['Sex', 'HasHypertension', 'Smokes', 'BlurredVision',
                        'FrequentHeadaches', 'MobilityDizziness', 'FamilyHistoryBrainEvent']

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

# Save model
joblib.dump(pipeline, "brain_model.pkl")
print("✅ Brain model trained and saved as brain_model.pkl")
