import boto3
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# Get the project root directory (go up one level from dbt_pipeline)
project_root = Path(__file__).parent.parent
bucket = os.getenv("S3_BUCKET_NAME")
file_path = project_root / "data" / "processed" / "fct_patient_features.parquet"

print(f"Uploading {file_path} to S3...")

if not file_path.exists():
    print(f"Error: File not found at {file_path}")
else:
    try:
        s3.upload_file(str(file_path), bucket, "processed/fct_patient_features.parquet")
        print("Uploaded successfully.")
    except Exception as e:
        print(f"Error uploading file: {e}")



