import joblib
import shap
import pandas as pd
from pathlib import Path

# Get paths relative to this file
ML_DIR = Path(__file__).parent
PROJECT_ROOT = ML_DIR.parent.parent
MODEL_DIR = ML_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

model = joblib.load(MODEL_DIR / "readmission_model.pkl")
df = pd.read_parquet(DATA_DIR / "fct_patient_features.parquet")
X = pd.get_dummies(df.drop(columns=["encounter_id", "readmitted_within_30d", "PATIENT", "days_to_next"]))

explainer = shap.Explainer(model)
shap_values = explainer(X)

# Save for inference use
joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

# Optional: SHAP summary plot (uncomment to view)
shap.summary_plot(shap_values, X, show=False)
