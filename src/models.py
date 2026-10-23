import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from src.encoding import ID_COL, FeatureEncoder
from src.features import CLASSES

K = len(CLASSES)
DEFAULT_FLOOR = 1e-3
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 40

GLOBAL_PARAMS = dict(max_depth=4, min_child_weight=10)
PER_PITCHER_PARAMS = dict(max_depth=3, min_child_weight=5)

def base_params(seed: int = 0) -> dict:
    return dict(
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
    )

def apply_floor(p: np.ndarray, floor: float) -> np.ndarray:
    p = np.maximum(p, floor)
    return p / p.sum(axis=1, keepdims=True)

class XGBPitchModel:
    """One multi-class XGBoost model on a set of rows (all pitchers, or one pitcher).

    fit(...) either
      - gets an eval_set and stops adding trees once that set stops improving
        (used while tuning on the validation season), or
      - gets n_estimators and builds exactly that many trees (used for the final refit).
    """

    def __init__(self, params: dict | None = None, use_pitcher_id: bool = True,
                 floor: float = DEFAULT_FLOOR, seed: int = 0):
        self.params = dict(params or {})
        self.use_pitcher_id = use_pitcher_id
        self.floor = floor
        self.seed = seed

    def fit(self, feats: pd.DataFrame, y: pd.Series, eval_set=None, n_estimators=None):
        y = pd.Series(y).reset_index(drop=True)
        self.enc_ = FeatureEncoder(self.use_pitcher_id).fit(feats)
        X = self.enc_.transform(feats)

        # XGBoost needs labels 0..m-1 with every one present, so a pitcher who never
        # threw some pitch type is trained on only the types he did throw.
        self.present_ = [c for c in CLASSES if c in set(y)]
        idx = {c: i for i, c in enumerate(self.present_)}
        yi = y.map(idx).to_numpy()

        if len(self.present_) < 2:
            self.const_ = np.array([1.0])
            self.best_n_ = 0
            return self
        self.const_ = None

        kwargs = {**base_params(self.seed), **self.params}
        if len(self.present_) == 2:
            # XGBoost's multi-class mode needs 3+ classes; two classes is a binary problem.
            kwargs["objective"] = "binary:logistic"
            kwargs["eval_metric"] = "logloss"
        if eval_set is not None:
            Xv_feats, yv = eval_set
            yv = pd.Series(yv).reset_index(drop=True)
            keep = yv.isin(self.present_).to_numpy()  
            # keep it a DataFrame (with column names) so it matches the training table
            Xv = self.enc_.transform(Xv_feats).iloc[np.flatnonzero(keep)]
            model = XGBClassifier(n_estimators=MAX_TREES,
                                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, **kwargs)
            model.fit(X, yi, eval_set=[(Xv, yv[keep].map(idx).to_numpy())], verbose=False)
            self.best_n_ = int(model.best_iteration) + 1
        else:
            if n_estimators is None:
                raise ValueError("fit needs either eval_set or n_estimators")
            model = XGBClassifier(n_estimators=int(n_estimators), **kwargs)
            model.fit(X, yi, verbose=False)
            self.best_n_ = int(n_estimators)
        self.model_ = model
        return self

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        X = self.enc_.transform(feats)
        if self.const_ is not None:
            present = np.tile(self.const_, (len(X), 1))
        else:
            present = self.model_.predict_proba(X, iteration_range=(0, self.best_n_))
        full = np.zeros((len(X), K))
        for j, c in enumerate(self.present_):
            full[:, CLASSES.index(c)] = present[:, j]
        return apply_floor(full, self.floor)

class PerPitcherXGB:
    """A separate XGBPitchModel for each pitcher (no pitcher-ID columns)."""

    def __init__(self, params: dict | None = None, floor: float = DEFAULT_FLOOR, seed: int = 0):
        self.params = dict(params or {})
        self.floor = floor
        self.seed = seed

    def fit(self, feats: pd.DataFrame, y: pd.Series, eval_feats=None, eval_y=None,
            n_estimators: dict | None = None):
        feats = feats.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.models_ = {}
        for pid in sorted(feats[ID_COL].unique()):
            m = (feats[ID_COL] == pid).to_numpy()
            model = XGBPitchModel(self.params, use_pitcher_id=False,
                                  floor=self.floor, seed=self.seed)
            if eval_feats is not None:
                ef = eval_feats.reset_index(drop=True)
                ey = pd.Series(eval_y).reset_index(drop=True)
                em = (ef[ID_COL] == pid).to_numpy()
                model.fit(feats[m], y[m], eval_set=(ef[em], ey[em]))
            else:
                model.fit(feats[m], y[m], n_estimators=n_estimators[int(pid)])
            self.models_[int(pid)] = model
        return self

    @property
    def best_n_(self) -> dict:
        return {pid: m.best_n_ for pid, m in self.models_.items()}

    def set_floor(self, floor: float) -> None:
        self.floor = floor
        for m in self.models_.values():
            m.floor = floor

    def predict_proba(self, feats: pd.DataFrame) -> np.ndarray:
        feats = feats.reset_index(drop=True)
        unknown = set(int(p) for p in feats[ID_COL].unique()) - set(self.models_)
        if unknown:
            raise ValueError(f"no model for pitchers: {sorted(unknown)}")
        out = np.zeros((len(feats), K))
        for pid, model in self.models_.items():
            m = (feats[ID_COL] == pid).to_numpy()
            if m.any():
                out[m] = model.predict_proba(feats[m])
        return out