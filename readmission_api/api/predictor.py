import pandas as pd
from api.model_loader import model

def predict_risk(input_data: dict) -> float:
    df = pd.DataFrame([input_data])
    
    # Transform to match training data format
    # gender: "male" -> "M", "female" -> "F"
    # race: stays lowercase
    # Column names: gender -> GENDER, race -> RACE, age -> age
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'})
    
    df = df.rename(columns={'gender': 'GENDER', 'race': 'RACE'})
    
    # Drop days_to_next if present (not used in training)
    if 'days_to_next' in df.columns or 'DAYS_TO_NEXT' in df.columns:
        df = df.drop(columns=['days_to_next', 'DAYS_TO_NEXT'], errors='ignore')
    
    df = pd.get_dummies(df)

    # Ensure all expected columns are present
    missing_cols = set(model.booster_.feature_name()) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = df[model.booster_.feature_name()]
    risk = model.predict_proba(df)[0][1]
    return round(risk, 4)
