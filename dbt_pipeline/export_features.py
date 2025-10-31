import duckdb
from pathlib import Path

# Get the project root directory (go up one level from dbt_pipeline)
project_root = Path(__file__).parent.parent

# Connect to DuckDB database
db_path = project_root / "data" / "duckdb" / "diabetes.duckdb"
con = duckdb.connect(str(db_path))

# Query the feature table
df = con.execute("select * from fct_patient_features").fetchdf()

# Export to Parquet
output_path = project_root / "data" / "processed" / "fct_patient_features.parquet"
df.to_parquet(str(output_path), index=False)

print(f"Feature table exported to: {output_path}")
