import glob
import os
import pandas as pd

RAW_GLOB = "data/raw/*.parquet"
OUT_PATH = "data/clean/pitches.parquet"

# Pitch-type codes that are not real pitches
NON_PITCH = {"PO", "IN", "UN", "AB", "AS"}

# Target label: raw Statcast types kept separate, except CS (slow curve,
# ~0.03% of pooled 2023-24 pitches) is folded into CU.
LABEL_MAP = {"CS": "CU"}
EXPECTED_CLASSES = {"FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV"}

REQUIRED = [
    "game_pk", "game_date", "pitcher", "player_name", "batter",
    "stand", "p_throws", "home_team", "away_team",
    "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "home_score", "away_score", "bat_score", "fld_score",
    "at_bat_number", "pitch_number", "pitch_type",
    # Outcome columns of the pitch itself. Features may only use these
    # SHIFTED to the previous pitch, never the current row.
    "description", "type",
]
OPTIONAL = ["n_thruorder_pitcher"]

frames = []
for f in sorted(glob.glob(RAW_GLOB)):
    df = pd.read_parquet(f)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{f}: missing columns {missing}")
    keep = REQUIRED + [c for c in OPTIONAL if c in df.columns]
    frames.append(df[keep])

df = pd.concat(frames, ignore_index=True)
df["season"] = pd.to_datetime(df["game_date"]).dt.year
n_start = len(df)

missing_type = df["pitch_type"].isna()
non_pitch = df["pitch_type"].isin(NON_PITCH)
print(f"rows loaded:            {n_start}")
print(f"dropped, missing type:  {int(missing_type.sum())}")
print(f"dropped, non-pitch code: {int(non_pitch.sum())}")
print("missing type by pitcher (top 5):")
print(df[missing_type]["player_name"].value_counts().head(5).to_string())

df = df[~missing_type & ~non_pitch]
df = (df.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"])
        .reset_index(drop=True))

# Target label. Use this column (not pitch_type) for the label and for any
# previous-pitch features, so labels stay consistent everywhere.
df["target"] = df["pitch_type"].replace(LABEL_MAP)

found = set(df["target"].unique())
if found != EXPECTED_CLASSES:
    raise SystemExit(
        f"Unexpected target classes.\n  unexpected: {sorted(found - EXPECTED_CLASSES)}"
        f"\n  missing: {sorted(EXPECTED_CLASSES - found)}"
    )

dupes = df.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum()
print(f"\nduplicate pitch rows: {dupes}")
print(f"final rows: {len(df)}")
print(df.groupby("season").size().to_string())
print("\noptional columns present:", [c for c in OPTIONAL if c in df.columns])
print("\ntarget classes:")
print(df["target"].value_counts().to_string())

os.makedirs("data/clean", exist_ok=True)
df.to_parquet(OUT_PATH, index=False)