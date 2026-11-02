import json
import sys
from datetime import date
import pandas as pd
from src.dataset.features import build_features
from src.dataset.splits import split
from src.evaluation.evaluate import log_loss
from src.evaluation.final_eval import fit_global_model, load_configs
from src.modeling.persistence import save_model

OUT_DIR = "models/global_xgb"
TOLERANCE = 0.002

def main():
    with open("results/final_results.json") as f:
        final = json.load(f)
    saved = next(r["log_loss"] for r in final["summary"] if r["model"] == "Global XGBoost")

    dev, _ = load_configs()
    df = pd.read_parquet("data/clean/pitches.parquet")
    train, test = split(df, "final", confirm_final=True)
    model = fit_global_model(build_features(train), train["target"], dev)

    now = log_loss(test["target"], model.predict_proba(build_features(test)))
    print(f"global log loss now {now:.4f} | saved final result {saved:.4f}")
    if abs(now - saved) > TOLERANCE:
        sys.exit("Scores do not match the saved final result; not saving the model.")

    save_model(model, OUT_DIR, extra={
        "trained_on_seasons": [2023, 2024, 2025],
        "final_2026_log_loss": round(saved, 4),
        "saved_on": date.today().isoformat(),
    })
    print(f"saved to {OUT_DIR}")

if __name__ == "__main__":
    main()