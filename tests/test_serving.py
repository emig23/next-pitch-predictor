from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.dataset.features import build_features
from src.dataset.request import features_for_request
from src.modeling.models import GLOBAL_PARAMS, XGBPitchModel
from src.modeling.persistence import load_model, save_model
from tests.test_final import make_raw
from tests.test_models import make_data

DATA = Path("data/clean/pitches.parquet")
CURRENT_KEYS = ["pitcher", "batter", "at_bat_number", "pitch_number", "balls", "strikes", "outs_when_up",
                "inning", "inning_topbot", "on_1b", "on_2b", "on_3b", "bat_score", "fld_score",
                "stand", "p_throws"]

def request_for_row(game: pd.DataFrame, i: int):
    """Turn row i of a game into (current, history) the way the API would receive it"""
    row = game.iloc[i]
    current = {k: row[k] for k in CURRENT_KEYS}
    for k in ("on_1b", "on_2b", "on_3b"):
        current[k] = bool(pd.notna(row[k]))
    history = [dict(at_bat_number=int(r.at_bat_number), pitch_number=int(r.pitch_number),
                    batter=int(r.batter),
                    pitch_type=r.target, description=r.description)
               for r in game.iloc[:i].itertuples()]
    return current, history

def check_game(game: pd.DataFrame):
    full = build_features(game)
    for i in range(len(game)):
        current, history = request_for_row(game, i)
        got = features_for_request(current, history)
        pd.testing.assert_frame_equal(got, full.iloc[[i]].reset_index(drop=True),
                                      check_exact=False)

def test_request_features_match_offline_features_on_a_whole_game():
    raw = make_raw(seasons=(2023,), pitchers=(11,), games=2)
    for _, game in raw.groupby("game_pk"):
        check_game(game.reset_index(drop=True))

@pytest.mark.skipif(not DATA.exists(), reason="clean data not available")
def test_request_features_match_offline_features_on_real_games():
    df = pd.read_parquet(DATA)
    for pid in df["pitcher"].unique()[:2]:
        d = df[df["pitcher"] == pid]
        game_pk = d["game_pk"].iloc[0]
        check_game(d[d["game_pk"] == game_pk].reset_index(drop=True))

def test_request_rejects_missing_fields_and_bad_input():
    raw = make_raw(seasons=(2023,), pitchers=(11,), games=1)
    current, history = request_for_row(raw, 5)
    bad = dict(current)
    bad.pop("balls")
    with pytest.raises(ValueError, match="missing"):
        features_for_request(bad, history)
    with pytest.raises(ValueError, match="unknown pitch_type"):
        features_for_request(current, [dict(history[0], pitch_type="ZZ")])
    with pytest.raises(ValueError, match="at_bat_number"):
        features_for_request(current, list(reversed(history)))

def test_first_pitch_of_game_has_empty_history_features():
    raw = make_raw(seasons=(2023,), pitchers=(11,), games=1)
    current, history = request_for_row(raw, 0)
    f = features_for_request(current, history)
    assert history == []
    assert f.loc[0, "pitches_so_far"] == 0
    assert f.loc[0, "prev_pitch_1"] == "NONE"
    assert f.filter(like="share_").isna().all(axis=None)

def test_saved_model_predicts_exactly_like_the_original(tmp_path):
    f, y = make_data(600, seed=0)
    model = XGBPitchModel(GLOBAL_PARAMS, use_pitcher_id=True).fit(f, y, n_estimators=25)
    save_model(model, tmp_path / "m", extra={"trained_on": [2023, 2024, 2025]})
    loaded = load_model(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict_proba(f), model.predict_proba(f), atol=1e-6)
    assert loaded.floor == model.floor and loaded.best_n_ == model.best_n_

def test_load_refuses_a_different_feature_layout(tmp_path):
    import json
    f, y = make_data(300, seed=1)
    save_model(XGBPitchModel(GLOBAL_PARAMS, use_pitcher_id=True).fit(f, y, n_estimators=5),
               tmp_path / "m")
    meta_path = tmp_path / "m" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["columns"] = meta["columns"][:-1]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="feature layout"):
        load_model(tmp_path / "m")

def test_fit_global_model_matches_the_model_used_in_the_final_evaluation():
    from src.evaluation.final_eval import fit_global_model, load_configs  # noqa: F401
    from tests.test_final import CFG_DEV
    f, y = make_data(400, seed=2)
    m = fit_global_model(f, y, CFG_DEV)
    assert m.best_n_ == CFG_DEV["global"]["n_estimators"]
    assert m.floor == CFG_DEV["global"]["floor"]