"""
NLP-based diagnosis prediction from chief complaint text.
"""
import joblib
from pathlib import Path
from typing import List, Dict

# Get the directory where this file is located
API_DIR = Path(__file__).parent.parent  # readmission_api/
MODEL_DIR = API_DIR / "ml" / "models"

# Lazy load models
_nlp_model = None
_diagnosis_labels = None

def get_nlp_model():
    global _nlp_model
    if _nlp_model is None:
        model_path = MODEL_DIR / "nlp_diagnosis_model.pkl"
        if model_path.exists():
            _nlp_model = joblib.load(model_path)
        else:
            raise FileNotFoundError(
                f"NLP diagnosis model not found at {model_path}. Run train_nlp_diagnosis.py first."
            )
    return _nlp_model

def get_diagnosis_labels():
    global _diagnosis_labels
    if _diagnosis_labels is None:
        labels_path = MODEL_DIR / "diagnosis_labels.pkl"
        if labels_path.exists():
            _diagnosis_labels = joblib.load(labels_path)
        else:
            _diagnosis_labels = []
    return _diagnosis_labels

def predict_diagnosis_from_complaint(complaint_text: str, top_k: int = 3) -> List[Dict[str, any]]:
    """
    Predict likely diagnoses from chief complaint text.
    
    Args:
        complaint_text: Patient's chief complaint
        top_k: Number of top predictions to return
        
    Returns:
        List of dicts with 'diagnosis' and 'probability'
    """
    if not complaint_text or len(complaint_text.strip()) < 3:
        return []
    
    try:
        model = get_nlp_model()
        
        # Get prediction probabilities
        proba = model.predict_proba([complaint_text])[0]
        classes = model.classes_
        
        # Get top K predictions
        top_indices = proba.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if proba[idx] > 0.05:  # Only return if >5% confidence
                results.append({
                    "diagnosis": classes[idx],
                    "probability": round(float(proba[idx]), 4)
                })
        
        return results
    except Exception as e:
        print(f"NLP prediction error: {e}")
        return []

