import numpy as np
import pandas as pd

from src.features import CLASSES

K = len(CLASSES)


class PitcherMixBaseline:
    """Baseline A. p(c) = (n_c + a) / (N + a*K), with a small add-a smoothing so a
    pitch type the pitcher never threw in training does not get probability 0
    (which would make log loss infinite). Argmax of this is the 'always predict
    the most common pitch' version, so top-1 accuracy covers that too"""

    def __init__(self, a: float = 0.5):
        if a <= 0:
            raise ValueError("a must be > 0")
        self.a = a

    def fit(self, train: pd.DataFrame):
        counts = (
            pd.crosstab(train["pitcher"], train["target"])
            .reindex(columns=CLASSES, fill_value=0)
        )
        self.mix_ = (counts + self.a).div(counts.sum(axis=1) + self.a * K, axis=0)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        unseen = set(df["pitcher"].unique()) - set(self.mix_.index)
        if unseen:
            raise ValueError(f"pitchers not in training data: {sorted(unseen)}")
        return self.mix_.loc[df["pitcher"]].to_numpy()


class CountBaseline:
    """Baseline B. p(c | pitcher, count) = (n_c,count + alpha * pA(c)) / (N_count + alpha).

    alpha is a pseudo-count strength: with few observations in a count the
    estimate stays near Baseline A, with many it follows the count's own
    frequencies. A count never seen in training falls back to Baseline A.
    alpha is tuned on the validation season."""

    def __init__(self, alpha: float = 10.0, a: float = 0.5):
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.alpha = alpha
        self.a = a

    def fit(self, train: pd.DataFrame):
        self.prior_ = PitcherMixBaseline(self.a).fit(train)
        self.counts_ = (
            train.groupby(["pitcher", "balls", "strikes", "target"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=CLASSES, fill_value=0)
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        prior = self.prior_.predict_proba(df)
        idx = pd.MultiIndex.from_frame(df[["pitcher", "balls", "strikes"]])
        c = self.counts_.reindex(idx).fillna(0).to_numpy(dtype=float)
        n = c.sum(axis=1, keepdims=True)
        return (c + self.alpha * prior) / (n + self.alpha)