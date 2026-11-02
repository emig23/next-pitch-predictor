import glob
import os
import pandas as pd

RAW_GLOB = "data/raw/*.parquet"
OUT_PATH = "data/clean/pitches.parquet"

# Pitch-type codes that are not real pitches
NON_PITCH = {"PO", "IN", "UN", "AB", "AS"}

LABEL_MAP = {"CS": "CU"}
EXPECTED_CLASSES = {"FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV"}

REQUIRED = [
    "game_pk", "game_date", "pitcher", "player_name", "batter",
    "stand", "p_throws", "home_team", "away_team",
    "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "home_score", "away_score", "bat_score", "fld_score",
    "at_bat_number", "pitch_number", "pitch_type", "description", "type",
]

OPTIONAL = ["n_thruorder_pitcher"]

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns we use. Raises ValueError if a required one is missing."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    return df[REQUIRED + [c for c in OPTIONAL if c in df.columns]]

def clean(df: pd.DataFrame, require_all_classes: bool = True, verbose: bool = False
          ) -> pd.DataFrame:
    """Raw table (columns from select_columns) -> cleaned pitch table.

    require_all_classes: the full rebuild checks that all 10 pitch types appear. A day of
    new games may not contain every type, so the ingestion job passes False; an
    UNEXPECTED type is an error either way."""
    df = df.copy()
    df["season"] = pd.to_datetime(df["game_date"]).dt.year
    n_start = len(df)

    missing_type = df["pitch_type"].isna()
    non_pitch = df["pitch_type"].isin(NON_PITCH)
    if verbose:
        print(f"rows loaded:            {n_start}")
        print(f"dropped, missing type:  {int(missing_type.sum())}")
        print(f"dropped, non-pitch code: {int(non_pitch.sum())}")
        print("missing type by pitcher (top 5):")
        print(df[missing_type]["player_name"].value_counts().head(5).to_string())

    df = df[~missing_type & ~non_pitch]
    df = (df.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"])
            .reset_index(drop=True))

    df["target"] = df["pitch_type"].replace(LABEL_MAP)

    found = set(df["target"].unique())
    unexpected = found - EXPECTED_CLASSES
    missing_classes = EXPECTED_CLASSES - found
    if unexpected or (require_all_classes and missing_classes):
        raise ValueError(
            f"Unexpected target classes.\n  unexpected: {sorted(unexpected)}"
            f"\n  missing: {sorted(missing_classes)}"
        )
    return df

def main():
    frames = []
    for f in sorted(glob.glob(RAW_GLOB)):
        try:
            frames.append(select_columns(pd.read_parquet(f)))
        except ValueError as e:
            raise SystemExit(f"{f}: {e}")

    try:
        df = clean(pd.concat(frames, ignore_index=True), verbose=True)
    except ValueError as e:
        raise SystemExit(str(e))

    dupes = df.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum()
    print(f"\nduplicate pitch rows: {dupes}")
    print(f"final rows: {len(df)}")
    print(df.groupby("season").size().to_string())
    print("\noptional columns present:", [c for c in OPTIONAL if c in df.columns])
    print("\ntarget classes:")
    print(df["target"].value_counts().to_string())

    os.makedirs("data/clean", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

if __name__ == "__main__":
    main()