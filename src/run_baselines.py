"""Dev-stage baselines: fit on 2023-24, tune alpha and evaluate on 2025.
Never touches 2026. Run from the project root:  python -m src.run_baselines
"""
import pandas as pd

from src.baselines import CountBaseline, PitcherMixBaseline
from src.evaluate import evaluate, per_pitcher
from src.splits import split

ALPHA_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]

df = pd.read_parquet("data/clean/pitches.parquet")
train, val = split(df, "dev")
print(f"train rows: {len(train)} (seasons {sorted(int(s) for s in train['season'].unique())})")
print(f"val rows:   {len(val)} (seasons {sorted(int(s) for s in val['season'].unique())})")

# Baseline A
base_a = PitcherMixBaseline().fit(train)
proba_a = base_a.predict_proba(val)
res_a = evaluate(val["target"], proba_a)

# Baseline B: choose alpha on validation log loss
rows = []
for alpha in ALPHA_GRID:
    p = CountBaseline(alpha=alpha).fit(train).predict_proba(val)
    rows.append({"alpha": alpha, **evaluate(val["target"], p)})
grid = pd.DataFrame(rows)
print("\nAlpha grid (Baseline B, validation):")
print(grid[["alpha", "log_loss", "top1", "top2"]].to_string(index=False, float_format="{:.4f}".format))

best_alpha = float(grid.loc[grid["log_loss"].idxmin(), "alpha"])
if best_alpha in (min(ALPHA_GRID), max(ALPHA_GRID)):
    print(f"\nWARNING: best alpha {best_alpha:g} is at the edge of the grid; widen the grid.")
proba_b = CountBaseline(alpha=best_alpha).fit(train).predict_proba(val)
res_b = evaluate(val["target"], proba_b)

summary = pd.DataFrame([
    {"model": "A: pitcher mix", **res_a},
    {"model": f"B: by count (alpha={best_alpha:g})", **res_b},
])
print("\nValidation (2025) results:")
print(summary.to_string(index=False, float_format="{:.4f}".format))

pa = per_pitcher(val, proba_a).set_index("name")
pb = per_pitcher(val, proba_b).set_index("name")
cmp_ = pd.DataFrame({
    "n": pa["n"],
    "logloss_A": pa["log_loss"],
    "logloss_B": pb["log_loss"],
})
cmp_["B_better"] = cmp_["logloss_B"] < cmp_["logloss_A"]
print("\nPer pitcher (validation):")
print(cmp_.to_string(float_format="{:.4f}".format))
print(f"\nB beats A for {int(cmp_['B_better'].sum())} of {len(cmp_)} pitchers")