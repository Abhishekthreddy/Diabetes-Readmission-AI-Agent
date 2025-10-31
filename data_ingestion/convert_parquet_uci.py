import pandas as pd
from pathlib import Path

raw_path = Path("data/raw/") / "diabetic_data.csv"
output_path = Path("data/processed/")
output_path.mkdir(parents=True, exist_ok=True)

print("Reading CSV...")
df = pd.read_csv(raw_path)

print("Writing Parquet...")
df.to_parquet(output_path / "uci_diabetes.parquet", index=False)
print("Done.")
