import json
import numpy as np
import pandas as pd
import pytest
from src import run_final
from src.evaluate import CLASSES
from src.final_eval import (bootstrap_by_game, check_criteria,
                            fit_all_and_predict, load_configs)
from src.splits import split

DESCS = ["ball", "called_strike", "swinging_strike", "foul", "hit_into_play"]

def make_raw(seed=0, seasons=(2023, 2024, 2025, 2026), pitchers=(11, 22, 33), games=6):
    """Raw pitch table with the real columns. The pitch depends on batter hand."""
    rng = np.random.default_rng(seed)
    rows, game_pk = [], 1000
    for season in seasons:
        for pitcher in pitchers:
            for _ in range(games):
                game_pk += 1
                ab = 1
                for pa in range(int(rng.integers(15, 22))):
                    stand = str(rng.choice(["L", "R"]))
                    balls = strikes = 0
                    for pn in range(1, int(rng.integers(2, 6)) + 1):
                        p = [.7, .2, .1] if stand == "L" else [.1, .2, .7]
                        t = str(rng.choice(["SL", "FF", "CH"], p=p))
                        rows.append(dict(
                            season=season, game_pk=game_pk, pitcher=pitcher,
                            player_name=f"P{pitcher}", batter=int(500 + pa % 9), stand=stand,
                            p_throws="R", inning=1 + pa // 6, inning_topbot="Top",
                            balls=balls, strikes=strikes, outs_when_up=pa % 3,
                            on_1b=999 if rng.random() < .3 else np.nan, on_2b=np.nan, on_3b=np.nan,
                            bat_score=0, fld_score=0, at_bat_number=ab, pitch_number=pn,
                            target=t, description=str(rng.choice(DESCS))))
                        balls = min(3, balls + int(rng.random() < .3))
                        strikes = min(2, strikes + int(rng.random() < .3))
                    ab += 1
    df = pd.DataFrame(rows)
    return df.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"]
                          ).reset_index(drop=True)

CFG_DEV = {
    "global": {"params": {"max_depth": 3, "min_child_weight": 5}, "n_estimators": 20,
               "floor": 1e-4},
    "per_pitcher": {"params": {"max_depth": 3, "min_child_weight": 5},
                    "n_estimators": {"11": 15, "22": 15, "33": 15}, "floor": 1e-3},
}
CFG_BASE = {"B": {"alpha": 20, "a": 0.5},
            "C": {"alpha": 20, "alpha_b": 20, "beta": 20, "a": 0.5}}

@pytest.fixture(scope="module")
def raw():
    return make_raw()

@pytest.fixture(scope="module")
def fitted(raw):
    train, test = split(raw, "final", confirm_final=True)
    return test, fit_all_and_predict(train, test, CFG_DEV, CFG_BASE)

def test_all_five_predictors_return_valid_probabilities(fitted):
    test, probas = fitted
    assert list(probas) == ["A", "B", "C", "Global XGBoost", "Per-pitcher XGBoost"]
    for p in probas.values():
        assert p.shape == (len(test), len(CLASSES))
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)

def test_final_fit_never_sees_the_test_season(raw):
    """Changing the 2026 labels must not change any prediction."""
    train, test = split(raw, "final", confirm_final=True)
    base = fit_all_and_predict(train, test, CFG_DEV, CFG_BASE)
    shuffled = test.copy()
    shuffled["target"] = np.random.default_rng(5).permutation(shuffled["target"].to_numpy())
    # features use earlier-pitch labels inside the test rows, so only baselines are exact
    again = fit_all_and_predict(train, shuffled, CFG_DEV, CFG_BASE)
    for m in ["A", "B", "C"]:
        assert np.array_equal(base[m], again[m])

def test_models_beat_uniform_where_signal_exists(fitted):
    test, probas = fitted
    from src.evaluate import log_loss
    assert log_loss(test["target"], probas["C"]) < log_loss(test["target"], probas["B"])

def test_bootstrap_intervals_contain_the_point_estimate(fitted):
    test, probas = fitted
    r = bootstrap_by_game(test, probas, n_boot=300, seed=1)
    for m in probas:
        lo, hi = r[m]["ci"]
        assert lo <= r[m]["log_loss"] <= hi
    lo, hi = r["C"]["gain_vs_B_ci"]
    assert lo <= r["C"]["gain_vs_B"] <= hi
    assert r["_n_games"] == test["game_pk"].nunique()

def test_bootstrap_is_reproducible_and_paired(fitted):
    test, probas = fitted
    a = bootstrap_by_game(test, probas, n_boot=200, seed=3)
    b = bootstrap_by_game(test, probas, n_boot=200, seed=3)
    assert a == b
    same = {"B": probas["B"], "C": probas["B"], "Global XGBoost": probas["B"]}
    r = bootstrap_by_game(test, same, n_boot=200, seed=3)
    assert r["C"]["gain_vs_B"] == 0 and r["C"]["gain_vs_B_ci"] == [0.0, 0.0]

def test_criteria_pass_and_fail_correctly():
    good = {"Global XGBoost": {"gain_vs_B": .1, "gain_vs_C": .05, "gain_vs_C_ci": [.02, .08]}}
    rows = check_criteria(good, {"Global XGBoost": 12}, 15)
    assert all(ok for _, ok, _ in rows)
    bad = {"Global XGBoost": {"gain_vs_B": .01, "gain_vs_C": -.01, "gain_vs_C_ci": [-.03, .01]}}
    rows = check_criteria(bad, {"Global XGBoost": 5}, 15)
    assert [ok for _, ok, _ in rows] == [True, False, False, False]

def test_run_final_requires_flag_and_refuses_overwrite(tmp_path, monkeypatch, raw):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run_final.main([])                      # no flag
    (tmp_path / "results").mkdir()
    (tmp_path / "results/final_results.json").write_text("{}")
    with pytest.raises(SystemExit):
        run_final.main(["--confirm-final"])     # already used

def test_run_final_end_to_end_writes_results_once(tmp_path, monkeypatch, raw):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/clean").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    raw.to_parquet("data/clean/pitches.parquet")
    json.dump(CFG_DEV, open("results/dev_config.json", "w"))
    json.dump(CFG_BASE, open("results/baselines_config.json", "w"))
    run_final.main(["--confirm-final"])
    out = json.load(open("results/final_results.json"))
    assert len(out["summary"]) == 5 and len(out["criteria"]) == 4
    with pytest.raises(SystemExit):
        run_final.main(["--confirm-final"])     # second run refused