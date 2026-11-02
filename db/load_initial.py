import sys
import pandas as pd
from db.connection import apply_schema, make_engine
from db.load import count_rows, load_clean

PARQUET = "data/clean/pitches.parquet"

def main():
    df = pd.read_parquet(PARQUET)
    engine = make_engine()
    apply_schema(engine)
    added = load_clean(engine, df)
    print("rows added   :", added)
    have = count_rows(engine)
    print("rows in tables:", have)
    if have["pitches"] < len(df):
        sys.exit(f"Row count mismatch: file has {len(df)} pitches, database has {have['pitches']}.")
    print(f"OK: database holds all {len(df)} pitches from {PARQUET}.")

if __name__ == "__main__":
    main()