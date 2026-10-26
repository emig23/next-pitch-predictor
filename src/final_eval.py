import json
import numpy as np
import pandas as pd
from src.baselines import CountBaseline, CountHandBaseline, PitcherMixBaseline
from src.evaluate import check_proba, labels_to_index
from src.features import build_features
from src.models import PerPitcherXGB, XGBPitchModel

MODEL_ORDER = ["A", "B", "C", "Global XGBoost", "Per-pitcher XGBoost"]

def load_configs(dev_path: str = "results/dev_config.json",
                 base_path: str = "results/baselines_config.json"):
    with open(dev_path) as f:
        dev = json.load(f)
    with open(base_path) as f:
        base = json.load(f)
    return dev, base

def fit_all_and_predict(train: pd.DataFrame, test: pd.DataFrame, dev: dict, base: dict) -> dict:
    """Refit every predictor on `train` with the saved settings; return {name: proba on test}."""
    b, c = base["B"], base["C"]
    out = {
        "A": PitcherMixBaseline(a=b["a"]).fit(train).predict_proba(test),
        "B": CountBaseline(alpha=b["alpha"], a=b["a"]).fit(train).predict_proba(test),
        "C": CountHandBaseline(alpha=c["alpha"], alpha_b=c["alpha_b"], beta=c["beta"],
                               a=c["a"]).fit(train).predict_proba(test),
    }
    ftr, fte = build_features(train), build_features(test)
    ytr = train["target"]

    g = dev["global"]
    gm = XGBPitchModel(g["params"], use_pitcher_id=True, floor=g["floor"]
                       ).fit(ftr, ytr, n_estimators=int(g["n_estimators"]))
    out["Global XGBoost"] = gm.predict_proba(fte)

    p = dev["per_pitcher"]
    trees = {int(k): int(v) for k, v in p["n_estimators"].items()}
    pm = PerPitcherXGB(p["params"], floor=p["floor"]).fit(ftr, ytr, n_estimators=trees)
    out["Per-pitcher XGBoost"] = pm.predict_proba(fte)

    for name, proba in out.items():
        check_proba(proba, len(test))
    return out

def per_pitch_loss(y, proba: np.ndarray) -> np.ndarray:
    idx = labels_to_index(y)
    return -np.log(np.clip(proba[np.arange(len(idx)), idx], 1e-12, 1.0))

def bootstrap_by_game(test: pd.DataFrame, probas: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% intervals for each model's log loss and for its gain over B and C.

    Pitches in one game are not independent, so whole games are resampled, not
    single pitches. Every model is scored on the same resampled games, so the
    gains are paired."""
    test = test.reset_index(drop=True)
    games = test["game_pk"].to_numpy()
    uniq, inv = np.unique(games, return_inverse=True)
    n_games = len(uniq)
    counts = np.bincount(inv, minlength=n_games).astype(float)
    sums = {m: np.bincount(inv, weights=per_pitch_loss(test["target"], p), minlength=n_games)
            for m, p in probas.items()}

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_games, size=(n_boot, n_games))
    denom = counts[draws].sum(axis=1)
    boot = {m: s[draws].sum(axis=1) / denom for m, s in sums.items()}

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    res = {}
    for m in probas:
        point = float(sums[m].sum() / counts.sum())
        res[m] = {"log_loss": point, "ci": ci(boot[m])}
        for ref in ("B", "C"):
            if m != ref:
                res[m][f"gain_vs_{ref}"] = float(sums[ref].sum() / counts.sum() - point)
                res[m][f"gain_vs_{ref}_ci"] = ci(boot[ref] - boot[m])
    res["_n_games"] = int(n_games)
    return res

def check_criteria(boot: dict, per_pitcher_wins: dict, n_pitchers: int) -> list:
    """Success criteria as (name, passed, detail)"""

    g = boot["Global XGBoost"]
    rows = [
        ("[SPEC] Global model beats Baseline B on log loss",
         g["gain_vs_B"] > 0, f"gain {g['gain_vs_B']:+.4f}"),
        ("[extra] Global model beats Baseline C on log loss (the fair bar)",
         g["gain_vs_C"] > 0, f"gain {g['gain_vs_C']:+.4f}"),
        ("[extra] Gain over C is clearly positive (95% interval above 0)",
         g["gain_vs_C_ci"][0] > 0,
         f"interval [{g['gain_vs_C_ci'][0]:+.4f}, {g['gain_vs_C_ci'][1]:+.4f}]"),
        ("[extra] Global model beats B for a majority of pitchers",
         per_pitcher_wins["Global XGBoost"] > n_pitchers / 2,
         f"{per_pitcher_wins['Global XGBoost']} of {n_pitchers}"),
    ]
    return rows