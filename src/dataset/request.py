import numpy as np
import pandas as pd
from src.dataset.features import CLASSES, FEATURE_COLUMNS, build_features

CURRENT_FIELDS = [
    "pitcher", "batter", "at_bat_number", "pitch_number", "balls", "strikes", "outs_when_up", "inning",
    "inning_topbot", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "stand", "p_throws",
]
HISTORY_FIELDS = ["at_bat_number", "pitch_number", "batter", "pitch_type", "description"]
GAME_PK = 0          # one game only, so any constant works
PLACEHOLDER = "FF"   # label of the pending pitch

def features_for_request(current: dict, history: list) -> pd.DataFrame:
    """current : the situation before the pitch (CURRENT_FIELDS; on_1b/2b/3b are True/False).
    history : earlier pitches this game, oldest first, each with HISTORY_FIELDS.
    Returns a one-row DataFrame with exactly FEATURE_COLUMNS."""
    missing = [k for k in CURRENT_FIELDS if k not in current]
    if missing:
        raise ValueError(f"features_for_request: current is missing {missing}")
    for i, h in enumerate(history):
        miss = [k for k in HISTORY_FIELDS if k not in h]
        if miss:
            raise ValueError(f"features_for_request: history[{i}] is missing {miss}")
        if h["pitch_type"] not in CLASSES:
            raise ValueError(f"features_for_request: history[{i}] has unknown pitch_type "
                             f"{h['pitch_type']!r}")

    ab_numbers = [h["at_bat_number"] for h in history] + [current["at_bat_number"]]
    if any(b < a for a, b in zip(ab_numbers, ab_numbers[1:])):
        raise ValueError("features_for_request: at_bat_number must not go down "
                         "(history must be oldest first)")

    base = {k: current[k] for k in CURRENT_FIELDS}
    for k in ("on_1b", "on_2b", "on_3b"):          # runner present -> any non-null value
        base[k] = 1.0 if current[k] else np.nan
    base.update(pitcher=int(current["pitcher"]), game_pk=GAME_PK)

    rows = [{**base, "at_bat_number": h["at_bat_number"], "pitch_number": h["pitch_number"],
             "batter": h["batter"], "target": h["pitch_type"], "description": h["description"]}
            for h in history]
    rows.append({**base, "target": PLACEHOLDER, "description": ""})

    feats = build_features(pd.DataFrame(rows))
    return feats.iloc[[-1]].reset_index(drop=True)[FEATURE_COLUMNS]