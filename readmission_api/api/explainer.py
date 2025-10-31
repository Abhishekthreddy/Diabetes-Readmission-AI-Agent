import pandas as pd
from api.model_loader import explainer, model

def get_shap_explanation(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])
    
    # Transform to match training data format (same as predictor.py)
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'male': 'M', 'female': 'F', 'm': 'M', 'f': 'F'})
    
    df = df.rename(columns={'gender': 'GENDER', 'race': 'RACE'})
    
    # Drop days_to_next if present (not used in training)
    if 'days_to_next' in df.columns or 'DAYS_TO_NEXT' in df.columns:
        df = df.drop(columns=['days_to_next', 'DAYS_TO_NEXT'], errors='ignore')
    
    df = pd.get_dummies(df)
    
    # Ensure all expected columns are present (same as predictor)
    missing_cols = set(model.booster_.feature_name()) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    
    # Reorder columns to match model's expected order
    df = df[model.booster_.feature_name()]
    
    shap_values = explainer(df)
    contribs = dict(zip(df.columns, shap_values.values[0]))
    return dict(sorted(contribs.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
