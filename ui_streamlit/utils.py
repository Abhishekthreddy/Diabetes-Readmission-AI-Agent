import requests
from config import API_BASE_URL

def predict(payload):
    """Legacy endpoint for structured risk prediction only"""
    url = f"{API_BASE_URL}/predict"
    r = requests.post(url, json=payload)
    print(f"Status: {r.status_code}")
    print(f"Response text: {r.text}")
    r.raise_for_status()
    return r.json()

def explain(payload):
    """Legacy endpoint for SHAP explanations only"""
    url = f"{API_BASE_URL}/explain"
    r = requests.post(url, json=payload)
    print(f"Status: {r.status_code}")
    print(f"Response text: {r.text}")
    r.raise_for_status()
    return r.json()

def predict_combined(payload):
    """
    Combined endpoint: Returns structured risk + SHAP + NLP diagnoses.
    Payload should include: age, gender, race, chief_complaint (optional)
    """
    url = f"{API_BASE_URL}/predict_combined"
    r = requests.post(url, json=payload)
    print(f"Status: {r.status_code}")
    print(f"Response text: {r.text}")
    r.raise_for_status()
    return r.json()