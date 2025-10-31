from pydantic import BaseModel
from typing import Optional, List, Dict

class PatientInput(BaseModel):
    age: int
    gender: str
    race: str
    chief_complaint: Optional[str] = None  # Optional text input for NLP

class PredictionOutput(BaseModel):
    readmission_risk: float

class ExplanationOutput(BaseModel):
    feature_contributions: dict

class DiagnosisOutput(BaseModel):
    diagnosis: str
    probability: float

class CombinedOutput(BaseModel):
    """Combined output from structured + NLP models"""
    readmission_risk: float
    risk_factors: Dict[str, float]  # SHAP values
    predicted_diagnoses: List[DiagnosisOutput]  # NLP predictions
