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

class CountHandBaseline:
    """Baseline C. Pitcher x batter hand x count, built in three steps.
 
    1. Start from Baseline B's answer for this pitcher and count.
    2. Adjust it for batter handedness: multiply each pitch type by
       (how often the pitcher throws it to this hand) / (how often overall),
       then rescale to sum to 1. This assumes the count effect and the hand
       effect multiply, which lets sparse cells still use both.
    3. Blend with the actual frequencies in this exact (pitcher, hand, count)
       cell, exactly like B does:  (n_c + alpha * step2) / (N + alpha).
 
    beta controls how far the 'how often to this hand' mix is pulled toward the
    pitcher's overall mix (larger = less hand effect). alpha_b is B's own
    smoothing and should be set to B's tuned value. alpha and beta are tuned on
    the validation season."""
 
    def __init__(self, alpha: float = 20.0, alpha_b: float = 100.0,
                 beta: float = 20.0, a: float = 0.5):
        if min(alpha, alpha_b, beta, a) <= 0:
            raise ValueError("alpha, alpha_b, beta and a must all be > 0")
        self.alpha, self.alpha_b, self.beta, self.a = alpha, alpha_b, beta, a
 
    @staticmethod
    def _check_stand(df: pd.DataFrame) -> None:
        bad = set(df["stand"].unique()) - {"L", "R"}
        if bad:
            raise ValueError(f"stand must be 'L' or 'R'; found {sorted(map(str, bad))}")
 
    def fit(self, train: pd.DataFrame):
        self._check_stand(train)
        self.b_ = CountBaseline(self.alpha_b, self.a).fit(train)
        self.hand_counts_ = (
            train.groupby(["pitcher", "stand", "target"]).size()
            .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
        )
        self.cell_counts_ = (
            train.groupby(["pitcher", "stand", "balls", "strikes", "target"]).size()
            .unstack(fill_value=0).reindex(columns=CLASSES, fill_value=0)
        )
        return self
 
    def _prior(self, df: pd.DataFrame) -> np.ndarray:
        pa = self.b_.prior_.predict_proba(df)               # Baseline A
        pb = self.b_.predict_proba(df)                      # Baseline B
        idx = pd.MultiIndex.from_frame(df[["pitcher", "stand"]])
        hc = self.hand_counts_.reindex(idx).fillna(0).to_numpy(dtype=float)
        hand_mix = (hc + self.beta * pa) / (hc.sum(axis=1, keepdims=True) + self.beta)
        prior = pb * (hand_mix / pa)
        return prior / prior.sum(axis=1, keepdims=True)
 
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_stand(df)
        prior = self._prior(df)
        idx = pd.MultiIndex.from_frame(df[["pitcher", "stand", "balls", "strikes"]])
        c = self.cell_counts_.reindex(idx).fillna(0).to_numpy(dtype=float)
        n = c.sum(axis=1, keepdims=True)
        return (c + self.alpha * prior) / (n + self.alpha)