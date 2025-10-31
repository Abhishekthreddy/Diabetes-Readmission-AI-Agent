import boto3
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bucket = os.getenv("S3_BUCKET_NAME")

# Upload UCI Diabetes file
uci_file = Path("data/processed/uci_diabetes.parquet")
if not uci_file.exists():
    print(f"❌ Error: {uci_file} not found")
else:
    print(f"Uploading {uci_file} to S3...")
    try:
        s3.upload_file(str(uci_file), bucket, "processed/uci_diabetes.parquet")
        print("✅ UCI Diabetes file uploaded successfully.")
    except Exception as e:
        print(f"❌ Error uploading UCI Diabetes file: {e}")

# Upload Synthea files
source_dir = Path("data/processed/synthea/")
if not source_dir.exists():
    print(f"❌ Error: {source_dir} directory not found")
else:
    parquet_files = list(source_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in {source_dir}")
    else:
        for file in parquet_files:
            s3_key = f"processed/synthea/{file.name}"
            print(f"Uploading {file.name} to S3 as {s3_key}...")
            try:
                s3.upload_file(str(file), bucket, s3_key)
                print(f"✅ {file.name} uploaded successfully.")
            except Exception as e:
                print(f"❌ Error uploading {file.name}: {e}")
        
print("\n✅ Upload process completed.")

#Upload fct_patient_features.parquet
file_path = Path("data/processed/fct_patient_features.parquet")

print(f"Uploading {file_path} to S3...")

if not file_path.exists():
    print(f"❌ Error: File not found at {file_path}")
else:
    try:
        s3.upload_file(str(file_path), bucket, "processed/fct_patient_features.parquet")
        print("✅ fct_patient_features.parquet uploaded successfully.")
    except Exception as e:
        print(f"❌ Error uploading fct_patient_features.parquet: {e}")


#Upload models to S3

model_files = [
    ("ml/models/readmission_model.pkl", "models/readmission_model.pkl"),
    ("ml/models/shap_explainer.pkl", "models/shap_explainer.pkl"),
]

for local_path, s3_key in model_files:
    print(f"Uploading {s3_key}...")
    s3.upload_file(local_path, bucket, s3_key)
    print(f"✅ {s3_key} uploaded successfully.")
    