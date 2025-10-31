# 🏥 Diabetes Readmission AI Agent

A hybrid AI system combining **structured machine learning** (LightGBM) and **natural language processing** (NLP) to predict diabetes patient readmission risk and infer likely diagnoses from chief complaints.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Features

- **📊 Readmission Risk Prediction**: LightGBM model predicts 30-day readmission probability
- **🩺 NLP Diagnosis Inference**: TF-IDF + Naive Bayes predicts likely diagnoses from chief complaints
- **📈 SHAP Explanations**: Interpretable AI with top risk factor contributions
- **🚀 REST API**: FastAPI endpoints deployed on AWS Lambda
- **🎨 Interactive UI**: Streamlit web interface for real-time predictions
- **🔄 Data Pipeline**: DBT models for feature engineering
- **☁️ Cloud-Ready**: Containerized deployment via AWS SAM

---

## 📊 Model Performance

| Model | Algorithm | Accuracy/Metric | Features |
|-------|-----------|-----------------|----------|
| Readmission Risk | LightGBM | ROC AUC (MLflow) | Age, Gender, Race |
| NLP Diagnosis | TF-IDF + Naive Bayes | 73.7% | 1000 TF-IDF features |
| SHAP Explainer | TreeExplainer | Top 5 factors | Same as risk model |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                            │
│              (Patient Data + Chief Complaint)                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  API Gateway + Lambda                        │
│                 POST /predict_combined                       │
└───────┬─────────────────────────────┬───────────────────────┘
        │                             │
        ▼                             ▼
┌──────────────────┐         ┌──────────────────────┐
│  Structured ML   │         │   NLP Diagnosis      │
│   (LightGBM)     │         │  (TF-IDF + NB)       │
│                  │         │                      │
│ • Age            │         │ • Chief Complaint    │
│ • Gender         │         │   (free text)        │
│ • Race           │         │                      │
│                  │         │ Output:              │
│ Output:          │         │ • Top 3 diagnoses    │
│ • Risk score     │         │ • Confidence scores  │
│ • SHAP values    │         │                      │
└──────────────────┘         └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker (for AWS deployment)
- AWS CLI configured (for deployment)
- Virtual environment

### Installation

```bash
# Clone the repository
git clone https://github.com/Abhishekthreddy/Diabetes-Readmission-AI-Agent.git
cd Diabetes-Readmission-AI-Agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r readmission_api/requirements.txt
pip install -r requirments.txt  # UI dependencies
```

### Data Setup

```bash
# Place your Synthea CSV files in data/raw/Synthea/
# Then convert to Parquet format
python data_ingestion/convert_synthea_parquet.py

# Run DBT pipeline to create feature tables
cd dbt_pipeline/diabetes_agent
dbt run
cd ../..
```

### Train Models

```bash
cd readmission_api

# Train structured risk model
python ml/train_model.py

# Train SHAP explainer
python ml/explain_model.py

# Train NLP diagnosis model
python ml/train_nlp_diagnosis.py

cd ..
```

### Run Locally

**API Server:**
```bash
cd readmission_api
uvicorn api.main:app --reload --port 8000
```

**Streamlit UI:**
```bash
cd ui_streamlit
streamlit run app.py
```

**Test Combined Workflow:**
```bash
python test_combined_workflow.py
```

---

## 📡 API Endpoints

### Combined Endpoint (Recommended)

**POST** `/predict_combined`

```json
{
  "age": 65,
  "gender": "male",
  "race": "white",
  "chief_complaint": "chest pain and shortness of breath"
}
```

**Response:**
```json
{
  "readmission_risk": 0.3988,
  "risk_factors": {
    "age": -0.320,
    "RACE_white": 0.075,
    "GENDER_M": 0.042
  },
  "predicted_diagnoses": [
    {
      "diagnosis": "Chronic congestive heart failure (disorder)",
      "probability": 0.8523
    }
  ]
}
```

### Legacy Endpoints

- **POST** `/predict` - Structured risk only
- **POST** `/explain` - SHAP explanations only
- **GET** `/` - API health check
- **GET** `/ping` - Ping endpoint

---

## ☁️ AWS Deployment

### Build and Deploy

```bash
# Build Docker image
sam build --use-container --template template.yaml

# Deploy to AWS (first time)
sam deploy --guided

# Deploy (subsequent times)
sam deploy
```

### Configuration

Update `ui_streamlit/config.py` with your API Gateway URL:
```python
API_BASE_URL = "https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/Prod"
```

---

## 📁 Project Structure

```
project-diabetes-ai-agent/
├── data/                          # Data storage
│   ├── raw/                       # Raw Synthea CSV files
│   └── processed/                 # Processed Parquet files
├── data_ingestion/                # Data conversion scripts
├── dbt_pipeline/                  # DBT data transformations
│   └── diabetes_agent/
│       └── models/
│           ├── staging/           # Staging models
│           └── marts/             # Feature tables
├── readmission_api/               # Main API application
│   ├── api/                       # FastAPI endpoints
│   │   ├── main.py               # API app + routes
│   │   ├── predictor.py          # Risk prediction
│   │   ├── explainer.py          # SHAP explanations
│   │   ├── nlp_predictor.py      # NLP diagnosis
│   │   └── schemas.py            # Pydantic models
│   ├── ml/                        # ML training scripts
│   │   ├── train_model.py        # Train LightGBM
│   │   ├── train_nlp_diagnosis.py # Train NLP
│   │   ├── explain_model.py      # Train SHAP
│   │   └── models/               # Trained models (.pkl)
│   ├── Dockerfile                # Lambda container
│   └── requirements.txt          # Python dependencies
├── ui_streamlit/                  # Streamlit UI
│   ├── app.py                    # Main UI
│   ├── utils.py                  # API client
│   └── config.py                 # API URL config
├── template.yaml                  # AWS SAM template
├── test_combined_workflow.py      # Integration tests
└── README.md                      # This file
```

---

## 🔧 Development

### Running Tests

```bash
# Test combined workflow
python test_combined_workflow.py

# Test API endpoints
curl -X POST "http://localhost:8000/predict_combined" \
  -H "Content-Type: application/json" \
  -d '{"age": 65, "gender": "male", "race": "white", "chief_complaint": "chest pain"}'
```

### Code Structure

- **Path Resolution**: All paths use `Path(__file__)` for portability
- **Lazy Loading**: Models load on-demand for Lambda optimization
- **Type Safety**: Pydantic schemas for request/response validation
- **Logging**: Comprehensive logging via Python logging module

---

## 📚 Documentation

- **[PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)** - Comprehensive project overview
- **[WORKFLOW_IMPLEMENTATION.md](WORKFLOW_IMPLEMENTATION.md)** - Technical implementation details
- **[PATH_REFERENCE.md](PATH_REFERENCE.md)** - Path resolution guide

---

## 🛠️ Tech Stack

### Backend
- **Python 3.12** - Programming language
- **FastAPI** - REST API framework
- **LightGBM** - Gradient boosting for risk prediction
- **scikit-learn** - NLP pipeline (TF-IDF + Naive Bayes)
- **SHAP** - Model explainability
- **pandas** - Data manipulation

### Data Pipeline
- **DBT** - Data transformation
- **DuckDB** - Analytics database
- **Parquet** - Columnar storage format

### ML Ops
- **MLflow** - Experiment tracking
- **joblib** - Model serialization

### Deployment
- **AWS Lambda** - Serverless compute
- **API Gateway** - HTTP API
- **Docker** - Containerization
- **AWS SAM** - Infrastructure as code

### Frontend
- **Streamlit** - Interactive web UI
- **requests** - HTTP client

---

## 📊 Data Sources

This project uses **Synthea** synthetic healthcare data:
- Patient demographics
- Encounter records with reason codes/descriptions
- Conditions and diagnoses

**Note**: Data files are not included in the repository due to size. Generate your own using [Synthea](https://github.com/synthetichealth/synthea).

---

## 🔐 Environment Variables

Create a `.env` file for local development:

```env
# API Configuration
API_BASE_URL=http://localhost:8000

# AWS Configuration (for deployment)
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=your-account-id

# MLflow (optional)
MLFLOW_TRACKING_URI=file:readmission_api/ml/mlflow_logs
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Synthea** - Synthetic healthcare data generation
- **FastAPI** - Modern web framework
- **LightGBM** - High-performance gradient boosting
- **SHAP** - Model interpretability

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🗺️ Roadmap

### Current Version (v1.0)
- ✅ Structured risk prediction
- ✅ NLP diagnosis inference
- ✅ Combined API endpoint
- ✅ Streamlit UI
- ✅ AWS Lambda deployment

### Future Enhancements
- [ ] Add more structured features (medications, lab results)
- [ ] Fine-tune BERT/BioBERT for medical NLP
- [ ] Implement RAG with medical knowledge base
- [ ] Multi-label classification for comorbidities
- [ ] Model versioning and A/B testing
- [ ] Real-time monitoring and alerting
- [ ] Automated retraining pipeline

---

**Built with ❤️ for better healthcare outcomes**

