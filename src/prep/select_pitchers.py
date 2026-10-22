import glob
import pandas as pd

MIN_PITCHES = 1500
MIN_START_SHARE = 0.80
N_FINAL = 15

df = pd.concat([pd.read_parquet(f) for f in glob.glob("data/selection/*.parquet")],
               ignore_index=True)
df["season"] = pd.to_datetime(df["game_date"]).dt.year
df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"])

# Starter = pitcher who threw their team's first pitch of the game.
# Top of an inning means the home team is pitching, bottom means the away team.
first = (df.groupby(["game_pk", "inning_topbot"], sort=False)
           .head(1)[["game_pk", "inning_topbot", "pitcher"]]
           .rename(columns={"pitcher": "starter"}))
df = df.merge(first, on=["game_pk", "inning_topbot"])
df["is_starter"] = df["pitcher"] == df["starter"]

# Appearances and starts per pitcher
games = (df.groupby(["pitcher", "game_pk"])["is_starter"].max().reset_index())
apps = games.groupby("pitcher").agg(appearances=("game_pk", "count"),
                                    starts=("is_starter", "sum"))
apps["start_share"] = apps["starts"] / apps["appearances"]

# Pitches per season (rows with a pitch type)
pitches = (df[df["pitch_type"].notna()]
           .groupby(["pitcher", "season"]).size().unstack(fill_value=0))
pitches["total_pitches"] = pitches.sum(axis=1)
pitches["min_season"] = pitches.drop(columns="total_pitches").min(axis=1)

names = df.groupby("pitcher")["player_name"].agg(lambda s: s.mode().iloc[0])

table = pitches.join(apps).join(names)
qualifying = table[(table["min_season"] >= MIN_PITCHES)
                   & (table["start_share"] >= MIN_START_SHARE)]
qualifying = qualifying.sort_values("total_pitches", ascending=False)
qualifying["rank"] = range(1, len(qualifying) + 1)

print(f"{len(qualifying)} pitchers qualify")
cols = ["rank", "player_name", 2023, 2024, 2025, "total_pitches", "start_share"]
print(qualifying[cols].to_string())

qualifying.to_csv("data/qualifying_pitchers.csv")
print(f"\nTop {N_FINAL} saved as the final list (pending arsenal review in Step 5).")