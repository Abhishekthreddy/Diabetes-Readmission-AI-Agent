"""
Train NLP model for diagnosis prediction from chief complaint text.
Uses Synthea's reasonDescription as training data.
"""
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Get paths relative to this file
ML_DIR = Path(__file__).parent
PROJECT_ROOT = ML_DIR.parent.parent
MODEL_DIR = ML_DIR / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load Synthea encounters with reason descriptions
encounters = pd.read_parquet(DATA_DIR / "synthea" / "encounters.parquet")

# Filter to rows with valid reason descriptions
df = encounters[['REASONDESCRIPTION', 'DESCRIPTION']].dropna()
df = df[df['REASONDESCRIPTION'].str.len() > 5]  # Remove very short text

print(f"Total training samples: {len(df)}")
print(f"Unique diagnoses: {df['REASONDESCRIPTION'].nunique()}")

# Use top N most common diagnoses for classification
top_n = 20
top_diagnoses = df['REASONDESCRIPTION'].value_counts().head(top_n).index.tolist()
df_filtered = df[df['REASONDESCRIPTION'].isin(top_diagnoses)]

print(f"\nFiltered to top {top_n} diagnoses: {len(df_filtered)} samples")
print("\nTop diagnoses:")
for i, diag in enumerate(top_diagnoses[:10], 1):
    count = (df_filtered['REASONDESCRIPTION'] == diag).sum()
    print(f"  {i}. {diag}: {count} samples")

# Prepare data
X = df_filtered['DESCRIPTION'].values  # Encounter description as input
y = df_filtered['REASONDESCRIPTION'].values  # Reason as target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Build pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

print("\nTraining NLP diagnosis model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_DIR / "nlp_diagnosis_model.pkl")
joblib.dump(top_diagnoses, MODEL_DIR / "diagnosis_labels.pkl")

print(f"\n✅ NLP diagnosis model saved to {MODEL_DIR / 'nlp_diagnosis_model.pkl'}")
print(f"✅ Diagnosis labels saved to {MODEL_DIR / 'diagnosis_labels.pkl'}")

