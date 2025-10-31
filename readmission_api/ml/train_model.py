import pandas as pd
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from pathlib import Path

# Get paths relative to this file
ML_DIR = Path(__file__).parent
PROJECT_ROOT = ML_DIR.parent.parent
MODEL_DIR = ML_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load data
df = pd.read_parquet(DATA_DIR / "fct_patient_features.parquet")
df = df.dropna()  # drop incomplete rows for simplicity

X = df.drop(columns=["encounter_id", "readmitted_within_30d", "PATIENT", "days_to_next"])
y = df["readmitted_within_30d"]

# Encode categorical
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# MLflow logging
mlflow.set_tracking_uri(f"file:{ML_DIR / 'mlflow_logs'}")
mlflow.set_experiment("Readmission Risk")

with mlflow.start_run():
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(model, "model")

    print(f"ROC AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model locally
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "readmission_model.pkl")
