import numpy as np
import pandas as pd
from src.features import CLASSES, FEATURE_COLUMNS

PREV_COLS = ["prev_pitch_1", "prev_pitch_2", "prev_pitch_3"]
PREV_CATS = CLASSES + ["NONE"]
RESULT_COL = "prev_result"
RESULT_CATS = ["NONE", "ball", "called_strike", "swinging_strike", "foul", "other"]
ID_COL = "pitcher_id"

NUMERIC_COLS = [c for c in FEATURE_COLUMNS if c not in PREV_COLS + [RESULT_COL, ID_COL]]

class FeatureEncoder:
    """use_pitcher_id=True  -> global model (adds one 0/1 column per training pitcher)
       use_pitcher_id=False -> per-pitcher model (pitcher columns left out)"""

    def __init__(self, use_pitcher_id: bool):
        self.use_pitcher_id = use_pitcher_id

    def fit(self, feats: pd.DataFrame):
        self.pitchers_ = sorted(int(p) for p in feats[ID_COL].unique()) if self.use_pitcher_id else []
        self.columns_ = (
            NUMERIC_COLS
            + [f"{c}={v}" for c in PREV_COLS for v in PREV_CATS]
            + [f"{RESULT_COL}={v}" for v in RESULT_CATS]
            + [f"pitcher={p}" for p in self.pitchers_]
        )
        return self

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLUMNS if c not in feats.columns]
        if missing:
            raise ValueError(f"encoder: missing feature columns {missing}")

        out = {c: feats[c].to_numpy(dtype=np.float32) for c in NUMERIC_COLS}
        for c in PREV_COLS:
            self._check(feats[c], PREV_CATS, c)
            for v in PREV_CATS:
                out[f"{c}={v}"] = (feats[c] == v).to_numpy(dtype=np.float32)
        self._check(feats[RESULT_COL], RESULT_CATS, RESULT_COL)
        for v in RESULT_CATS:
            out[f"{RESULT_COL}={v}"] = (feats[RESULT_COL] == v).to_numpy(dtype=np.float32)
        # A pitcher not seen at fit time simply gets all zeros in these columns.
        for p in self.pitchers_:
            out[f"pitcher={p}"] = (feats[ID_COL] == p).to_numpy(dtype=np.float32)
        return pd.DataFrame(out, index=feats.index)[self.columns_]

    @staticmethod
    def _check(series: pd.Series, allowed: list, name: str) -> None:
        bad = set(series.unique()) - set(allowed)
        if bad:
            raise ValueError(f"encoder: unexpected values in {name}: {sorted(map(str, bad))}")