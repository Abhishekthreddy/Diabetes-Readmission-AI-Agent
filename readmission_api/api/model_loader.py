import joblib
from pathlib import Path
import os

# Get the directory where this file is located
API_DIR = Path(__file__).parent.parent  # readmission_api/
MODEL_DIR = API_DIR / "ml" / "models"

model = joblib.load(MODEL_DIR / "readmission_model.pkl")
explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
