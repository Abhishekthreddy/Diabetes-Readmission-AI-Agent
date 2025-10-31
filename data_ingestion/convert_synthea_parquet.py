import pandas as pd
from pathlib import Path

input_dir = Path("data/raw/Synthea/")
output_dir = Path("data/processed/synthea")
output_dir.mkdir(parents=True, exist_ok=True)

files = ["patients", "encounters", "conditions"]

for fname in files:
    print(f"Processing {fname}.csv...")
    df = pd.read_csv(input_dir / f"{fname}.csv")
    df.to_parquet(output_dir / f"{fname}.parquet", index=False)

print("✅ All Synthea files converted to Parquet.")
