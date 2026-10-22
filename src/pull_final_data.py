import os
import time
import pandas as pd
from pybaseball import statcast_pitcher, cache

cache.enable()

OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

SEASONS = [2023, 2024, 2025, 2026]
pitchers = pd.read_csv("data/final_pitchers.csv")

for _, p in pitchers.iterrows():
    for season in SEASONS:
        path = f"{OUT_DIR}/{int(p['pitcher'])}_{season}.parquet"
        if os.path.exists(path):
            print("skip", path)
            continue
        print("pulling", p["player_name"], season)
        df = statcast_pitcher(f"{season}-03-15", f"{season}-10-05", int(p["pitcher"]))
        if df is None or df.empty:
            print("  no data")
            continue
        df = df[df["game_type"] == "R"]
        df.to_parquet(path, index=False)
        time.sleep(1)