import json
import os

import pandas as pd

from src.baselines import CountBaseline, PitcherMixBaseline
from src.calibration import reliability_table
from src.evaluate import evaluate, per_pitcher
from src.features import build_features
from src.models import GLOBAL_PARAMS, PER_PITCHER_PARAMS, PerPitcherXGB, XGBPitchModel
from src.splits import split

ALPHA_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]
GLOBAL_GRID = [dict(max_depth=d, min_child_weight=w) for d in (3, 5) for w in (5, 20)]
FLOOR_GRID = [1e-4, 1e-3, 1e-2]

df = pd.read_parquet("data/clean/pitches.parquet")
train, val = split(df, "dev")
ftr, fva = build_features(train), build_features(val)
ytr, yva = train["target"], val["target"]
print(f"train rows: {len(train)} | val rows: {len(val)}")

# ---------- baselines ----------
proba_a = PitcherMixBaseline().fit(train).predict_proba(val)
alpha_ll = {a: evaluate(yva, CountBaseline(alpha=a).fit(train).predict_proba(val))["log_loss"]
            for a in ALPHA_GRID}
best_alpha = min(alpha_ll, key=alpha_ll.get)
proba_b = CountBaseline(alpha=best_alpha).fit(train).predict_proba(val)
print(f"Baseline B best alpha: {best_alpha}")

# ---------- global model: small settings grid ----------
print("\nGlobal XGBoost settings (validation log loss):")
candidates = []
for params in GLOBAL_GRID:
    m = XGBPitchModel(params, use_pitcher_id=True).fit(ftr, ytr, eval_set=(fva, yva))
    ll = evaluate(yva, m.predict_proba(fva))["log_loss"]
    print(f"  {params}  trees={m.best_n_}  log_loss={ll:.4f}")
    candidates.append((ll, params, m))
_, best_global_params, global_model = min(candidates, key=lambda c: c[0])
print(f"  -> chose {best_global_params}")

# ---------- per-pitcher models ----------
print("\nTraining one model per pitcher ...")
pp_model = PerPitcherXGB(PER_PITCHER_PARAMS).fit(ftr, ytr, eval_feats=fva, eval_y=yva)

# ---------- probability floor ----------
def pick_floor(set_floor, predict):
    scores = {}
    for f in FLOOR_GRID:
        set_floor(f)
        scores[f] = evaluate(yva, predict())["log_loss"]
    best = min(scores, key=scores.get)
    set_floor(best)
    return best, scores


def _set_global_floor(f):
    global_model.floor = f


g_floor, g_scores = pick_floor(_set_global_floor, lambda: global_model.predict_proba(fva))
p_floor, p_scores = pick_floor(pp_model.set_floor, lambda: pp_model.predict_proba(fva))
print(f"\nfloor chosen: global {g_floor:g} | per-pitcher {p_floor:g}")

proba_g = global_model.predict_proba(fva)
proba_p = pp_model.predict_proba(fva)

# ---------- comparison ----------
models = {
    "A: pitcher mix": proba_a,
    f"B: by count (alpha={best_alpha})": proba_b,
    "Global XGBoost": proba_g,
    "Per-pitcher XGBoost": proba_p,
}
rows = []
for name, p in models.items():
    r = evaluate(yva, p)
    r["ece"] = reliability_table(yva, p)[1]
    rows.append({"model": name, **r})
summary = pd.DataFrame(rows)
print("\nValidation (2025) results:")
print(summary.to_string(index=False, float_format="{:.4f}".format))

b_ll = summary.loc[1, "log_loss"]
for i in (2, 3):
    gain = b_ll - summary.loc[i, "log_loss"]
    print(f"{summary.loc[i, 'model']}: log loss {gain:+.4f} vs B ({gain / b_ll:+.2%})")

per = {name: per_pitcher(val, p).set_index("name") for name, p in models.items()}
cmp_ = pd.DataFrame({
    "n": per["A: pitcher mix"]["n"],
    "A": per["A: pitcher mix"]["log_loss"],
    "B": per[f"B: by count (alpha={best_alpha})"]["log_loss"],
    "global": per["Global XGBoost"]["log_loss"],
    "per_pitcher": per["Per-pitcher XGBoost"]["log_loss"],
})
cmp_["global<B"] = cmp_["global"] < cmp_["B"]
cmp_["per_pitcher<B"] = cmp_["per_pitcher"] < cmp_["B"]
print("\nPer pitcher (validation log loss):")
print(cmp_.to_string(float_format="{:.4f}".format))
print(f"\nGlobal beats B for {int(cmp_['global<B'].sum())} of {len(cmp_)} pitchers")
print(f"Per-pitcher beats B for {int(cmp_['per_pitcher<B'].sum())} of {len(cmp_)} pitchers")

# ---------- calibration of the best model ----------
best_name = summary.loc[[2, 3], "log_loss"].idxmin()
best_name = summary.loc[best_name, "model"]
table, ece = reliability_table(yva, models[best_name])
print(f"\nCalibration of {best_name} (ECE {ece:.4f}):")
print(table.to_string(index=False, float_format="{:.3f}".format))

# ---------- save the chosen settings for the final refit ----------
config = {
    "baseline_B": {"alpha": best_alpha, "a": 0.5},
    "global": {
        "params": best_global_params,
        "n_estimators": int(global_model.best_n_),
        "floor": g_floor,
        "val_log_loss": float(summary.loc[2, "log_loss"]),
    },
    "per_pitcher": {
        "params": PER_PITCHER_PARAMS,
        "n_estimators": {str(k): int(v) for k, v in pp_model.best_n_.items()},
        "floor": p_floor,
        "val_log_loss": float(summary.loc[3, "log_loss"]),
    },
}
os.makedirs("results", exist_ok=True)
with open("results/dev_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("\nsaved results/dev_config.json")