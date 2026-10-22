import os
import pandas as pd
from pybaseball import statcast, cache

cache.enable()

OUT_DIR = "data/selection"
os.makedirs(OUT_DIR, exist_ok=True)

SEASONS = [2023, 2024, 2025]  # 2026 is the test season, so it is not used for selection
KEEP = ["game_pk", "game_date", "game_type", "pitcher", "player_name",
        "pitch_type", "inning", "inning_topbot", "at_bat_number", "pitch_number"]


def month_windows(year):
    cur = pd.Timestamp(f"{year}-03-15")
    end = pd.Timestamp(f"{year}-10-05")
    while cur <= end:
        stop = min(cur + pd.offsets.MonthEnd(0), end)
        yield cur, stop
        cur = stop + pd.Timedelta(days=1)


for year in SEASONS:
    for start, stop in month_windows(year):
        path = f"{OUT_DIR}/{year}_{start:%m%d}.parquet"
        if os.path.exists(path):
            print("skip", path)
            continue
        print("pulling", start.date(), "to", stop.date())
        df = statcast(start_dt=f"{start:%Y-%m-%d}", end_dt=f"{stop:%Y-%m-%d}")
        if df is None or df.empty:
            continue
        df = df[df["game_type"] == "R"]  # regular season only
        missing = set(KEEP) - set(df.columns)
        if missing:
            raise SystemExit(f"Missing columns: {missing}")
        df[KEEP].to_parquet(path, index=False)