import argparse
import json
import os
import sys
import pandas as pd
from src.evaluate import evaluate, per_pitcher
from src.final_eval import (MODEL_ORDER, bootstrap_by_game, check_criteria,
                            fit_all_and_predict, load_configs)
from src.splits import split

OUT_JSON = "results/final_results.json"
OUT_CSV = "results/final_per_pitcher.csv"

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-final", action="store_true",
                    help="required: you are sure the model and settings are locked")
    args = ap.parse_args(argv)
    if not args.confirm_final:
        sys.exit("Refusing to run: pass --confirm-final only when everything is locked and committed.")
    if os.path.exists(OUT_JSON):
        sys.exit(f"Refusing to run: {OUT_JSON} already exists. The test season is used once.")

    dev, base = load_configs()
    df = pd.read_parquet("data/clean/pitches.parquet")
    train, test = split(df, "final", confirm_final=True)
    print(f"train rows {len(train)} (seasons {sorted(int(s) for s in train['season'].unique())}) | "
          f"test rows {len(test)} (seasons {sorted(int(s) for s in test['season'].unique())})")

    probas = fit_all_and_predict(train, test, dev, base)
    ytest = test["target"]

    summary = pd.DataFrame([{"model": m, **evaluate(ytest, probas[m])} for m in MODEL_ORDER])
    print("\nTEST (2026) results:")
    print(summary.to_string(index=False, float_format="{:.4f}".format))

    boot = bootstrap_by_game(test, probas)
    print(f"\n95% intervals (resampling {boot['_n_games']} games):")
    for m in MODEL_ORDER:
        r = boot[m]
        line = f"  {m:<22} log loss {r['log_loss']:.4f}  [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        if "gain_vs_C" in r:
            line += f"   gain vs C {r['gain_vs_C']:+.4f} [{r['gain_vs_C_ci'][0]:+.4f}, {r['gain_vs_C_ci'][1]:+.4f}]"
        print(line)

    per = {m: per_pitcher(test, probas[m]).set_index("name") for m in MODEL_ORDER}
    table = pd.DataFrame({"n": per["A"]["n"]})
    for m in MODEL_ORDER:
        table[m] = per[m]["log_loss"]
    wins = {m: int((table[m] < table["B"]).sum()) for m in MODEL_ORDER}
    print("\nPer pitcher (2026 log loss):")
    print(table.to_string(float_format="{:.4f}".format))
    print(f"\nBeats B for: " + ", ".join(f"{m} {wins[m]}/{len(table)}" for m in MODEL_ORDER if m != "B"))

    print("\nSuccess criteria:")
    criteria = check_criteria(boot, wins, len(table))
    for name, ok, detail in criteria:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} ({detail})")

    os.makedirs("results", exist_ok=True)
    table.to_csv(OUT_CSV)
    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary.to_dict("records"), "bootstrap": boot,
                   "criteria": [{"name": n, "passed": bool(ok), "detail": d} for n, ok, d in criteria]},
                  f, indent=2)
    print(f"\nsaved {OUT_JSON} and {OUT_CSV}. Report these numbers as they are.")

if __name__ == "__main__":
    main()