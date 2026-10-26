import numpy as np
import pandas as pd
from src.final_eval import per_pitch_loss
from src.features import CLASSES

SHOWN = ["B", "C", "Global XGBoost"]
MIN_STARTER_PITCHES = 60
OPTIONAL = ["game_date", "home_team", "away_team"]

def pick_games(test: pd.DataFrame, global_proba: np.ndarray, per_pitcher: int = 2):
    test = test.reset_index(drop=True)
    loss = per_pitch_loss(test["target"], global_proba)
    g = (pd.DataFrame({"pitcher": test["pitcher"], "game_pk": test["game_pk"], "loss": loss})
         .groupby(["pitcher", "game_pk"]).agg(n=("loss", "size"), loss=("loss", "mean"))
         .reset_index())
    picks = []
    for pid, grp in g.groupby("pitcher"):
        big = grp[grp["n"] >= MIN_STARTER_PITCHES]
        pool = (big if len(big) >= per_pitcher else grp).sort_values(["loss", "game_pk"])
        m = len(pool)
        idx = sorted({min(m - 1, (m * (k + 1)) // (per_pitcher + 1)) for k in range(per_pitcher)})
        picks += [(int(pid), int(pool.iloc[i]["game_pk"])) for i in idx]
    return picks

def build_demo(test: pd.DataFrame, probas: dict, per_pitcher: int = 2) -> dict:
    test = test.reset_index(drop=True)
    picks = pick_games(test, probas["Global XGBoost"], per_pitcher)
    games = []
    for pid, gpk in picks:
        mask = ((test["pitcher"] == pid) & (test["game_pk"] == gpk)).to_numpy()
        rows = np.flatnonzero(mask)
        d = test.iloc[rows]
        pitches = []
        for r in rows:
            x = test.iloc[r]
            pitches.append({
                "ab": int(x["at_bat_number"]), "pn": int(x["pitch_number"]),
                "inning": int(x["inning"]), "top": int(x["inning_topbot"] == "Top"),
                "outs": int(x["outs_when_up"]), "balls": int(x["balls"]),
                "strikes": int(x["strikes"]),
                "on": [int(pd.notna(x["on_1b"])), int(pd.notna(x["on_2b"])), int(pd.notna(x["on_3b"]))],
                "diff": int(x["fld_score"] - x["bat_score"]),
                "stand": str(x["stand"]), "actual": str(x["target"]),
                "p": {m: [round(float(v), 4) for v in probas[m][r]] for m in SHOWN},
            })
        info = {"pitcher": pid, "name": str(d["player_name"].iloc[0]),
                "throws": str(d["p_throws"].iloc[0]), "game_pk": gpk, "pitches": pitches}
        for c in OPTIONAL:
            if c in d.columns:
                info[c] = str(d[c].iloc[0])
        games.append(info)
    games.sort(key=lambda g: (g["name"], g["game_pk"]))
    return {"classes": CLASSES, "models": SHOWN, "games": games}