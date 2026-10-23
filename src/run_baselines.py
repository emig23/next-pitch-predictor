import json
import os
import pandas as pd
from src.baselines import CountBaseline, CountHandBaseline, PitcherMixBaseline
from src.evaluate import evaluate, per_pitcher
from src.splits import split

ALPHA_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500]      
C_ALPHA_GRID = [20, 50, 100, 200, 500, 1000, 2000, 5000]   
C_BETA_GRID = [20, 50, 200, 500, 1000, 2000]               

def edge_warning(name, value, grid):
    if value in (min(grid), max(grid)):
        print(f"WARNING: best {name} {value:g} is at the edge of its grid; consider widening it.")


df = pd.read_parquet("data/clean/pitches.parquet")
train, val = split(df, "dev")
yva = val["target"]
print(f"train rows: {len(train)} (seasons {sorted(int(s) for s in train['season'].unique())})")
print(f"val rows:   {len(val)} (seasons {sorted(int(s) for s in val['season'].unique())})")

# ---------- Baseline A ----------
proba_a = PitcherMixBaseline().fit(train).predict_proba(val)

# ---------- Baseline B ----------
rows = []
for alpha in ALPHA_GRID:
    p = CountBaseline(alpha=alpha).fit(train).predict_proba(val)
    rows.append({"alpha": alpha, **evaluate(yva, p)})
grid = pd.DataFrame(rows)
print("\nAlpha grid (Baseline B, validation):")
print(grid[["alpha", "log_loss", "top1", "top2"]].to_string(index=False, float_format="{:.4f}".format))
best_alpha = float(grid.loc[grid["log_loss"].idxmin(), "alpha"])
edge_warning("alpha", best_alpha, ALPHA_GRID)
proba_b = CountBaseline(alpha=best_alpha).fit(train).predict_proba(val)

# ---------- Baseline C ----------
rows = []
for alpha in C_ALPHA_GRID:
    for beta in C_BETA_GRID:
        p = CountHandBaseline(alpha=alpha, alpha_b=best_alpha, beta=beta).fit(train).predict_proba(val)
        rows.append({"alpha": alpha, "beta": beta, "log_loss": evaluate(yva, p)["log_loss"]})
cgrid = pd.DataFrame(rows)
print("\nBaseline C grid (validation log loss; rows = alpha, columns = beta):")
print(cgrid.pivot(index="alpha", columns="beta", values="log_loss")
      .to_string(float_format="{:.4f}".format))
best = cgrid.loc[cgrid["log_loss"].idxmin()]
best_c_alpha, best_c_beta = float(best["alpha"]), float(best["beta"])
edge_warning("C alpha", best_c_alpha, C_ALPHA_GRID)
edge_warning("C beta", best_c_beta, C_BETA_GRID)
proba_c = CountHandBaseline(alpha=best_c_alpha, alpha_b=best_alpha,
                            beta=best_c_beta).fit(train).predict_proba(val)

# ---------- results ----------
models = {
    "A: pitcher mix": proba_a,
    f"B: by count (alpha={best_alpha:g})": proba_b,
    f"C: count + batter hand (alpha={best_c_alpha:g}, beta={best_c_beta:g})": proba_c,
}
summary = pd.DataFrame([{"model": n, **evaluate(yva, p)} for n, p in models.items()])
print("\nValidation (2025) results:")
print(summary.to_string(index=False, float_format="{:.4f}".format))

# For context: the saved model scores from run_models (no retraining here)
if os.path.exists("results/dev_config.json"):
    with open("results/dev_config.json") as f:
        cfg = json.load(f)
    c_ll = summary.iloc[2]["log_loss"]
    for key, label in (("global", "Global XGBoost"), ("per_pitcher", "Per-pitcher XGBoost")):
        ll = cfg[key]["val_log_loss"]
        print(f"{label} (saved dev score) {ll:.4f}  ->  {c_ll - ll:+.4f} vs C, "
              f"{summary.iloc[1]['log_loss'] - ll:+.4f} vs B")

names = list(models)
per = {n: per_pitcher(val, p).set_index("name") for n, p in models.items()}
cmp_ = pd.DataFrame({
    "n": per[names[0]]["n"],
    "A": per[names[0]]["log_loss"],
    "B": per[names[1]]["log_loss"],
    "C": per[names[2]]["log_loss"],
})
cmp_["C<B"] = cmp_["C"] < cmp_["B"]
print("\nPer pitcher (validation log loss):")
print(cmp_.to_string(float_format="{:.4f}".format))
print(f"\nB beats A for {int((cmp_['B'] < cmp_['A']).sum())} of {len(cmp_)} pitchers")
print(f"C beats B for {int(cmp_['C<B'].sum())} of {len(cmp_)} pitchers")

os.makedirs("results", exist_ok=True)
with open("results/baselines_config.json", "w") as f:
    json.dump({"B": {"alpha": best_alpha, "a": 0.5},
               "C": {"alpha": best_c_alpha, "alpha_b": best_alpha,
                     "beta": best_c_beta, "a": 0.5}}, f, indent=2)
print("\nsaved results/baselines_config.json")