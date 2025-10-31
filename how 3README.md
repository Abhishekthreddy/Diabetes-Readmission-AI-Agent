[1mdiff --git a/README.md b/README.md[m
[1mnew file mode 100644[m
[1mindex 0000000..d00fcb5[m
[1m--- /dev/null[m
[1m+++ b/README.md[m
[36m@@ -0,0 +1,394 @@[m
[32m+[m[32m<<<<<<< HEAD[m
[32m+[m[32m# Diabetes-Readmission-AI-Agent[m
[32m+[m[32mA hybrid clinical co-pilot that leverages both machine learning and NLP to predict hospital readmission risk for diabetes patients, and infer likely diagnoses from chief complaints.[m
[32m+[m[32m=======[m
[32m+[m[32m# 🏥 Diabetes Readmission AI Agent[m
[32m+[m
[32m+[m[32mA hybrid AI system combining **structured machine learning** (LightGBM) and **natural language processing** (NLP) to predict diabetes patient readmission risk and infer likely diagnoses from chief complaints.[m
[32m+[m
[32m+[m[32m[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)[m
[32m+[m[32m[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)[m
[32m+[m[32m[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## 🎯 Features[m
[32m+[m
[32m+[m[32m- **📊 Readmission Risk Prediction**: LightGBM model predicts 30-day readmission probability[m
[32m+[m[32m- **🩺 NLP Diagnosis Inference**: TF-IDF + Naive Bayes predicts likely diagnoses from chief complaints[m
[32m+[m[32m- **📈 SHAP Explanations**: Interpretable AI with top risk factor contributions[m
[32m+[m[32m- **🚀 REST API**: FastAPI endpoints deployed on AWS Lambda[m
[32m+[m[32m- **🎨 Interactive UI**: Streamlit web interface for real-time predictions[m
[32m+[m[32m- **🔄 Data Pipeline**: DBT models for feature engineering[m
[32m+[m[32m- **☁️ Cloud-Ready**: Containerized deployment via AWS SAM[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## 📊 Model Performance[m
[32m+[m
[32m+[m[32m| Model | Algorithm | Accuracy/Metric | Features |[m
[32m+[m[32m|-------|-----------|-----------------|----------|[m
[32m+[m[32m| Readmission Risk | LightGBM | ROC AUC (MLflow) | Age, Gender, Race |[m
[32m+[m[32m| NLP Diagnosis | TF-IDF + Naive Bayes | 73.7% | 1000 TF-IDF features |[m
[32m+[m[32m| SHAP Explainer | TreeExplainer | Top 5 factors | Same as risk model |[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## 🏗️ Architecture[m
[32m+[m
[32m+[m[32m```[m
[32m+[m[32m┌─────────────────────────────────────────────────────────────┐[m
[32m+[m[32m│                      Streamlit UI                            │[m
[32m+[m[32m│              (Patient Data + Chief Complaint)                │[m
[32m+[m[32m└────────────────────────┬────────────────────────────────────┘[m
[32m+[m[32m                         │[m
[32m+[m[32m                         ▼[m
[32m+[m[32m┌─────────────────────────────────────────────────────────────┐[m
[32m+[m[32m│                  API Gateway + Lambda                        │[m
[32m+[m[32m│                 POST /predict_combined                       │[m
[32m+[m[32m└───────┬─────────────────────────────┬───────────────────────┘[m
[32m+[m[32m        │                             │[m
[32m+[m[32m        ▼                             ▼[m
[32m+[m[32m┌──────────────────┐         ┌──────────────────────┐[m
[32m+[m[32m│  Structured ML   │         │   NLP Diagnosis      │[m
[32m+[m[32m│   (LightGBM)     │         │  (TF-IDF + NB)       │[m
[32m+[m[32m│                  │         │                      │[m
[32m+[m[32m│ • Age            │         │ • Chief Complaint    │[m
[32m+[m[32m│ • Gender         │         │   (free text)        │[m
[32m+[m[32m│ • Race           │         │                      │[m
[32m+[m[32m│                  │         │ Output:              │[m
[32m+[m[32m│ Output:          │         │ • Top 3 diagnoses    │[m
[32m+[m[32m│ • Risk score     │         │ • Confidence scores  │[m
[32m+[m[32m│ • SHAP values    │         │                      │[m
[32m+[m[32m└──────────────────┘         └──────────────────────┘[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## 🚀 Quick Start[m
[32m+[m
[32m+[m[32m### Prerequisites[m
[32m+[m
[32m+[m[32m- Python 3.12+[m
[32m+[m[32m- Docker (for AWS deployment)[m
[32m+[m[32m- AWS CLI configured (for deployment)[m
[32m+[m[32m- Virtual environment[m
[32m+[m
[32m+[m[32m### Installation[m
[32m+[m
[32m+[m[32m```bash[m
[32m+[m[32m# Clone the repository[m
[32m+[m[32mgit clone https://github.com/yourusername/Diabetes-Readmission-AI-Agent.git[m
[32m+[m[32mcd Diabetes-Readmission-AI-Agent[m
[32m+[m
[32m+[m[32m# Create and activate virtual environment[m
[32m+[m[32mpython -m venv .venv[m
[32m+[m[32msource .venv/bin/activate  # On Windows: .venv\Scripts\activate[m
[32m+[m
[32m+[m[32m# Install dependencies[m
[32m+[m[32mpip install -r readmission_api/requirements.txt[m
[32m+[m[32mpip install -r requirments.txt  # UI dependencies[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Data Setup[m
[32m+[m
[32m+[m[32m```bash[m
[32m+[m[32m# Place your Synthea CSV files in data/raw/Synthea/[m
[32m+[m[32m# Then convert to Parquet format[m
[32m+[m[32mpython data_ingestion/convert_synthea_parquet.py[m
[32m+[m
[32m+[m[32m# Run DBT pipeline to create feature tables[m
[32m+[m[32mcd dbt_pipeline/diabetes_agent[m
[32m+[m[32mdbt run[m
[32m+[m[32mcd ../..[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Train Models[m
[32m+[m
[32m+[m[32m```bash[m
[32m+[m[32mcd readmission_api[m
[32m+[m
[32m+[m[32m# Train structured risk model[m
[32m+[m[32mpython ml/train_model.py[m
[32m+[m
[32m+[m[32m# Train SHAP explainer[m
[32m+[m[32mpython ml/explain_model.py[m
[32m+[m
[32m+[m[32m# Train NLP diagnosis model[m
[32m+[m[32mpython ml/train_nlp_diagnosis.py[m
[32m+[m
[32m+[m[32mcd ..[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m### Run Locally[m
[32m+[m
[32m+[m[32m**API Server:**[m
[32m+[m[32m```bash[m
[32m+[m[32mcd readmission_api[m
[32m+[m[32muvicorn api.main:app --reload --port 8000[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m**Streamlit UI:**[m
[32m+[m[32m```bash[m
[32m+[m[32mcd ui_streamlit[m
[32m+[m[32mstreamlit run app.py[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m**Test Combined Workflow:**[m
[32m+[m[32m```bash[m
[32m+[m[32mpython test_combined_workflow.py[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## 📡 API Endpoints[m
[32m+[m
[32m+[m[32m### Combined Endpoint (Recommended)[m
[32m+[m
[32m+[m[32m**POST** `/predict_combined`[m
[32m+[m
[32m+[m[32m```json[m
[32m+[m[32m{[m
[32m+[m[32m  "age": 65,[m
[32m+[m[32m  "gender": "male",[m
[32m+[m[32m  "race": "white",[m
[32m+[m[32m  "chief_complaint": "chest pain and shortness of breath"[m
[32m+[m[32m}[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32m**Response:**[m
[32m+[m[32m```json[m
[32m+[m[32m{[m
[32m+[m[32m  "readmission_risk": 0.3988,[m
[32m+[m[32m  "risk_factors": {[m
[32m+[m[32m    "age": -0.320,[m
[32m+[m[32m    "RACE_white": 0.075,[m
[32m+[m[32m    "GENDER_M": 0.042[m
[32m+[m[32m  },[m
[32