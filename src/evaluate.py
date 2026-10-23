import numpy as np
import pandas as pd
from src.features import CLASSES

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
EPS = 1e-12  

def labels_to_index(y) -> np.ndarray:
    y = pd.Series(y)
    unknown = set(y.unique()) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    return y.map(CLASS_INDEX).to_numpy()

def check_proba(proba: np.ndarray, n_rows: int) -> None:
    if proba.shape != (n_rows, len(CLASSES)):
        raise ValueError(f"proba shape {proba.shape}, expected {(n_rows, len(CLASSES))}")
    if not np.isfinite(proba).all():
        raise ValueError("proba contains NaN or inf")
    if (proba < 0).any():
        raise ValueError("proba contains negative values")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("proba rows do not sum to 1")

def log_loss(y, proba: np.ndarray) -> float:
    idx = labels_to_index(y)
    p_true = proba[np.arange(len(idx)), idx]
    return float(-np.mean(np.log(np.clip(p_true, EPS, 1.0))))

def top_k_accuracy(y, proba: np.ndarray, k: int) -> float:
    idx = labels_to_index(y)
    top_k = np.argsort(-proba, axis=1)[:, :k]
    return float((top_k == idx[:, None]).any(axis=1).mean())

def evaluate(y, proba: np.ndarray) -> dict:
    """Aggregate metrics. n_floored counts rows where the true class got
    (almost) zero probability, which is a sign a model needs a probability floor."""
    y = pd.Series(y).reset_index(drop=True)
    check_proba(proba, len(y))
    idx = labels_to_index(y)
    p_true = proba[np.arange(len(idx)), idx]
    return {
        "n": int(len(y)),
        "log_loss": log_loss(y, proba),
        "top1": top_k_accuracy(y, proba, 1),
        "top2": top_k_accuracy(y, proba, 2),
        "n_floored": int((p_true < EPS).sum()),
    }

def per_pitcher(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Same metrics, one row per pitcher. df must be the rows proba was made for."""
    df = df.reset_index(drop=True)
    check_proba(proba, len(df))
    rows = []
    for pid, g in df.groupby("pitcher"):
        m = evaluate(g["target"], proba[g.index.to_numpy()])
        m["pitcher"] = pid
        m["name"] = g["player_name"].iloc[0]
        rows.append(m)
    cols = ["pitcher", "name", "n", "log_loss", "top1", "top2", "n_floored"]
    return pd.DataFrame(rows)[cols]