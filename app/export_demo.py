import json
import sys
import pandas as pd
from app.demo_export import build_demo
from src.evaluate import log_loss
from src.final_eval import fit_all_and_predict, load_configs
from src.splits import split

TOLERANCE = 0.002

def main():
    with open("results/final_results.json") as f:
        final = json.load(f)
    saved = next(r["log_loss"] for r in final["summary"] if r["model"] == "Global XGBoost")

    dev, base = load_configs()
    df = pd.read_parquet("data/clean/pitches.parquet")
    train, test = split(df, "final", confirm_final=True)
    probas = fit_all_and_predict(train, test, dev, base)

    now = log_loss(test["target"], probas["Global XGBoost"])
    print(f"global log loss now {now:.4f} | saved final result {saved:.4f}")
    if abs(now - saved) > TOLERANCE:
        sys.exit("Scores do not match the saved final result; not writing the demo file.")

    demo = build_demo(test, probas)
    with open("app/demo_data.json", "w") as f:
        json.dump(demo, f, separators=(",", ":"))
    n = sum(len(g["pitches"]) for g in demo["games"])
    print(f"saved app/demo_data.json: {len(demo['games'])} games, {n} pitches")

if __name__ == "__main__":
    main()