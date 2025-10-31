from fastapi import FastAPI
from mangum import Mangum
from api.schemas import PatientInput, PredictionOutput, ExplanationOutput, CombinedOutput
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

app = FastAPI()

# Don't import these at module level - lazy load for Lambda cold start optimization
_predictor = None
_explainer = None
_nlp_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        logger.info("Loading predictor...")
        from api.predictor import predict_risk
        _predictor = predict_risk
        logger.info("Predictor loaded!")
    return _predictor

def get_explainer_func():
    global _explainer
    if _explainer is None:
        logger.info("Loading explainer...")
        from api.explainer import get_shap_explanation
        _explainer = get_shap_explanation
        logger.info("Explainer loaded!")
    return _explainer

def get_nlp_predictor():
    global _nlp_predictor
    if _nlp_predictor is None:
        logger.info("Loading NLP predictor...")
        from api.nlp_predictor import predict_diagnosis_from_complaint
        _nlp_predictor = predict_diagnosis_from_complaint
        logger.info("NLP predictor loaded!")
    return _nlp_predictor

@app.get("/")
def root():
    logger.info("Root endpoint hit")
    return {"message": "Diabetes Readmission API", "status": "running"}

@app.get("/ping")
def ping():
    logger.info("Ping endpoint hit")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: PatientInput):
    logger.info(f"Predict called with: {input.dict()}")
    try:
        predict_func = get_predictor()
        risk = predict_func(input.dict())
        return {"readmission_risk": risk}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise

@app.post("/explain", response_model=ExplanationOutput)
def explain(input: PatientInput):
    logger.info(f"Explain called with: {input.dict()}")
    try:
        explain_func = get_explainer_func()
        contribs = explain_func(input.dict())
        return {"feature_contributions": contribs}
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}", exc_info=True)
        raise

@app.post("/predict_combined", response_model=CombinedOutput)
def predict_combined(input: PatientInput):
    """
    Combined endpoint: Structured risk model + NLP diagnosis prediction.
    Returns readmission risk, SHAP explanations, and NLP-predicted diagnoses.
    """
    logger.info(f"Combined prediction called with: {input.dict()}")
    try:
        # Get structured model predictions
        predict_func = get_predictor()
        explain_func = get_explainer_func()
        
        risk = predict_func(input.dict())
        risk_factors = explain_func(input.dict())
        
        # Get NLP diagnosis predictions if complaint provided
        predicted_diagnoses = []
        if input.chief_complaint and len(input.chief_complaint.strip()) > 0:
            nlp_func = get_nlp_predictor()
            predicted_diagnoses = nlp_func(input.chief_complaint, top_k=3)
        
        return {
            "readmission_risk": risk,
            "risk_factors": risk_factors,
            "predicted_diagnoses": predicted_diagnoses
        }
    except Exception as e:
        logger.error(f"Combined prediction error: {str(e)}", exc_info=True)
        raise

# Lambda handler
handler = Mangum(app, lifespan="off")
