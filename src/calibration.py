import numpy as np
import pandas as pd
from src.evaluate import labels_to_index

BIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

def reliability_table(y, proba: np.ndarray):
    """Returns (table, ece). ece = average gap between predicted and observed,
    weighted by how many predictions fall in each bin. Lower is better."""
    idx = labels_to_index(y)
    n = len(idx)
    hit = np.zeros_like(proba)
    hit[np.arange(n), idx] = 1.0
    p, h = proba.ravel(), hit.ravel()

    bins = np.digitize(p, BIN_EDGES[1:-1])      # 0 .. len(BIN_EDGES)-2
    rows = []
    for b in range(len(BIN_EDGES) - 1):
        m = bins == b
        if not m.any():
            continue
        rows.append({
            "bin": f"{BIN_EDGES[b]:.2f}-{BIN_EDGES[b + 1]:.2f}",
            "n": int(m.sum()),
            "mean_predicted": float(p[m].mean()),
            "observed": float(h[m].mean()),
        })
    table = pd.DataFrame(rows)
    ece = float((table["n"] * (table["mean_predicted"] - table["observed"]).abs()).sum()
                / table["n"].sum())
    return table, ece