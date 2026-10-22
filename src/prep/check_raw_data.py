import glob
import pandas as pd

rows = []
for f in sorted(glob.glob("data/raw/*.parquet")):
    df = pd.read_parquet(f, columns=["pitcher", "player_name", "game_date",
                                     "game_type", "pitch_type"])
    pid, season = f.replace("\\", "/").split("/")[-1].replace(".parquet", "").split("_")
    rows.append({
        "pitcher": int(pid),
        "name": df["player_name"].mode().iloc[0],
        "season": int(season),
        "pitches": len(df),
        "missing_type": int(df["pitch_type"].isna().sum()),
        "first_game": df["game_date"].min(),
        "last_game": df["game_date"].max(),
        "non_regular": int((df["game_type"] != "R").sum()),
    })

summary = pd.DataFrame(rows)
print("files:", len(summary))
print("\nPitches per pitcher-season:")
print(summary.pivot(index="name", columns="season", values="pitches").to_string())
print("\n2026 date ranges:")
print(summary[summary["season"] == 2026][["name", "first_game", "last_game"]].to_string(index=False))
print("\nRows with non-regular game_type:", summary["non_regular"].sum())
print("Rows with missing pitch_type:", summary["missing_type"].sum())