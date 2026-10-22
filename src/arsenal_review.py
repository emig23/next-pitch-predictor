import glob
import pandas as pd

TOP_N = 15
HIGH, LOW = 0.10, 0.03

q = pd.read_csv("data/qualifying_pitchers.csv")
q = q.sort_values("rank")

df = pd.concat([pd.read_parquet(f) for f in glob.glob("data/selection/*.parquet")],
               ignore_index=True)
df = df[df["pitch_type"].notna()]
df["season"] = pd.to_datetime(df["game_date"]).dt.year

counts = (df.groupby(["pitcher", "season", "pitch_type"])
            .size().rename("n").reset_index())
counts["share"] = counts["n"] / counts.groupby(["pitcher", "season"])["n"].transform("sum")
mix = counts.pivot_table(index=["pitcher", "pitch_type"], columns="season",
                         values="share", fill_value=0)

def has_arsenal_change(pid):
    m = mix.loc[pid]
    return [t for t, row in m.iterrows() if row.max() >= HIGH and row.min() < LOW]

final, rejected = [], []
for _, r in q.iterrows():
    changed = has_arsenal_change(r["pitcher"])
    if changed:
        rejected.append((r["rank"], r["player_name"], changed))
    else:
        final.append(r)
    if len(final) == TOP_N:
        break

final = pd.DataFrame(final)
print("Rejected for arsenal change:")
for rank, name, types in rejected:
    print(f"  rank {rank}: {name} -> {types}")

print("\nFinal list:")
print(final[["rank", "player_name", "pitcher", "total_pitches"]].to_string(index=False))

final[["pitcher", "player_name", "rank"]].to_csv("data/final_pitchers.csv", index=False)

# Pitch mix per final pitcher, for you to read
pd.set_option("display.float_format", "{:.1%}".format)
for pid, name in zip(final["pitcher"], final["player_name"]):
    print(f"\n{name}")
    print(mix.loc[pid])