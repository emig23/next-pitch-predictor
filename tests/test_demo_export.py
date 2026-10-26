import json
import numpy as np
import pytest
from app.demo_export import SHOWN, build_demo, pick_games
from src.final_eval import fit_all_and_predict
from src.splits import split
from tests.test_final import CFG_BASE, CFG_DEV, make_raw

@pytest.fixture(scope="module")
def fitted():
    raw = make_raw(seed=4, games=8)
    train, test = split(raw, "final", confirm_final=True)
    return test.reset_index(drop=True), fit_all_and_predict(train, test, CFG_DEV, CFG_BASE)

def test_two_games_per_pitcher_and_rule_is_deterministic(fitted):
    test, probas = fitted
    a = pick_games(test, probas["Global XGBoost"])
    assert a == pick_games(test, probas["Global XGBoost"])
    assert len(a) == 2 * test["pitcher"].nunique()
    assert len(set(a)) == len(a)

def test_picks_are_never_the_best_or_worst_game(fitted):
    test, probas = fitted
    from src.final_eval import per_pitch_loss
    test = test.assign(loss=per_pitch_loss(test["target"], probas["Global XGBoost"]))
    for pid, gpk in pick_games(test, probas["Global XGBoost"]):
        g = test[test["pitcher"] == pid].groupby("game_pk")["loss"].mean().sort_values()
        assert g.index[0] != gpk and g.index[-1] != gpk

def test_demo_rows_match_the_source_rows(fitted):
    test, probas = fitted
    demo = build_demo(test, probas)
    assert demo["models"] == SHOWN
    for g in demo["games"]:
        rows = test[(test["pitcher"] == g["pitcher"]) & (test["game_pk"] == g["game_pk"])]
        assert len(rows) == len(g["pitches"])
        assert [p["actual"] for p in g["pitches"]] == rows["target"].tolist()
        for p in g["pitches"]:
            for m in SHOWN:
                assert len(p["p"][m]) == 10
                assert abs(sum(p["p"][m]) - 1) < 2e-3
    json.dumps(demo)   # must be JSON-serializable