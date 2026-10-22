"""Feature pipeline for next-pitch prediction.

Every feature is known BEFORE the pitch is thrown. Nothing here may use the
current pitch's type, result, velocity, spin or location.

Input: the clean DataFrame from clean_data.py, sorted by
pitcher, game_pk, at_bat_number, pitch_number (one row per pitch).
Output: a DataFrame with the same index and exactly FEATURE_COLUMNS.
"""
import pandas as pd

SORT_KEYS = ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
GAME_KEYS = ["pitcher", "game_pk"]
AB_KEYS = ["pitcher", "game_pk", "at_bat_number"]

CLASSES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV"]

# Collapse Statcast's many "description" values into a few outcomes.
# Only ever applied to the PREVIOUS pitch in the same at-bat.
RESULT_MAP = {
    "ball": "ball",
    "blocked_ball": "ball",
    "automatic_ball": "ball",
    "pitchout": "ball",
    "called_strike": "called_strike",
    "automatic_strike": "called_strike",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "foul_pitchout": "foul",
}

FEATURE_COLUMNS = (
    # Situation
    ["balls", "strikes", "outs", "inning",
     "runner_on_1b", "runner_on_2b", "runner_on_3b",
     "score_diff", "pitcher_is_home"]
    # Matchup
    + ["batter_left", "pitcher_left"]
    # At-bat sequence (reset every at-bat)
    + ["pitch_number", "prev_pitch_1", "prev_pitch_2", "prev_pitch_3", "prev_result"]
    # Game context
    + ["pitches_so_far"]
    + [f"share_{c}" for c in CLASSES]
    + ["times_through_order"]
    # Identity (global model only)
    + ["pitcher_id"]
)

REQUIRED_INPUT = [
    "pitcher", "game_pk", "batter", "at_bat_number", "pitch_number",
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
    "stand", "p_throws", "target", "description",
]
NUMERIC_INPUT = ["balls", "strikes", "outs_when_up", "inning",
                 "bat_score", "fld_score", "pitch_number", "at_bat_number"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INPUT if c not in df.columns]
    if missing:
        raise ValueError(f"build_features: missing input columns {missing}")

    nan_cols = [c for c in NUMERIC_INPUT if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"build_features: NaN values in {nan_cols}")

    # The sequence features rely on row order, so refuse unsorted input.
    if not df.sort_values(SORT_KEYS, kind="stable").index.equals(df.index):
        raise ValueError(f"build_features: input must be sorted by {SORT_KEYS}")

    out = pd.DataFrame(index=df.index)

    # ---- Situation (state of the game before the pitch) ----
    out["balls"] = df["balls"].astype("int64")
    out["strikes"] = df["strikes"].astype("int64")
    out["outs"] = df["outs_when_up"].astype("int64")
    out["inning"] = df["inning"].astype("int64")
    out["runner_on_1b"] = df["on_1b"].notna().astype("int64")
    out["runner_on_2b"] = df["on_2b"].notna().astype("int64")
    out["runner_on_3b"] = df["on_3b"].notna().astype("int64")
    # Pitcher's team is the fielding team, so positive = pitcher's team ahead.
    out["score_diff"] = (df["fld_score"] - df["bat_score"]).astype("int64")
    # Top of the inning means the away team bats, so the home team is pitching.
    out["pitcher_is_home"] = (df["inning_topbot"] == "Top").astype("int64")

    # ---- Matchup ----
    out["batter_left"] = (df["stand"] == "L").astype("int64")
    out["pitcher_left"] = (df["p_throws"] == "L").astype("int64")

    # ---- At-bat sequence: previous pitches in the SAME at-bat only ----
    out["pitch_number"] = df["pitch_number"].astype("int64")
    g_ab = df.groupby(AB_KEYS, sort=False)
    for k in (1, 2, 3):
        out[f"prev_pitch_{k}"] = g_ab["target"].shift(k).fillna("NONE")

    prev_desc = g_ab["description"].shift(1)
    result = prev_desc.map(RESULT_MAP).fillna("other")
    out["prev_result"] = result.where(prev_desc.notna(), "NONE")

    # ---- Game context: counts BEFORE the current pitch ----
    g_game_keys = [df["pitcher"], df["game_pk"]]
    so_far = df.groupby(GAME_KEYS, sort=False).cumcount()
    out["pitches_so_far"] = so_far.astype("int64")
    denom = so_far.where(so_far > 0)  # NaN on the first pitch of the game
    for c in CLASSES:
        is_c = (df["target"] == c).astype("int64")
        # cumsum includes the current row, so subtract it back out.
        before = is_c.groupby(g_game_keys, sort=False).cumsum() - is_c
        out[f"share_{c}"] = (before / denom).astype("float64")

    # Which time through the order this batter is facing the pitcher.
    # Dense rank only depends on at-bats up to the current one.
    out["times_through_order"] = (
        df.groupby(GAME_KEYS + ["batter"], sort=False)["at_bat_number"]
        .rank(method="dense")
        .astype("int64")
    )

    # ---- Identity ----
    out["pitcher_id"] = df["pitcher"].astype("int64")

    return out[FEATURE_COLUMNS]