from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from api.main import create_app
from db.connection import make_engine
from db.load import load_clean
from src.dataset.features import CLASSES, build_features
from src.modeling.models import GLOBAL_PARAMS, XGBPitchModel
from src.modeling.persistence import load_model, save_model
from tests.test_db import engine, make_clean, needs_db  # noqa: F401  (engine is a fixture)
from tests.test_final import make_raw

REAL_MODEL = Path("models/global_xgb")
REAL_DATA = Path("data/clean/pitches.parquet")

@pytest.fixture(autouse=True)
def api_ignores_the_env_database(monkeypatch):
    """The app looks for DATABASE_URL at startup. Without this, every test that builds an app
    would try to reach your real database (and wait if it is slow or down). Tests that need a
    database pass one in explicitly through create_app(engine=...)."""
    monkeypatch.setattr("api.main.find_database_url", lambda: None)

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    raw = make_raw(seasons=(2023,), pitchers=(11, 22, 33), games=3)
    model = XGBPitchModel(GLOBAL_PARAMS, use_pitcher_id=True).fit(
        build_features(raw), raw["target"], n_estimators=20)
    path = tmp_path_factory.mktemp("model") / "m"
    save_model(model, path, extra={"trained_on_seasons": [2023]})
    return raw, str(path)

@pytest.fixture
def client(trained):
    with TestClient(create_app(model_dir=trained[1])) as c:
        yield c

def api_request(game: pd.DataFrame, i: int) -> dict:
    """The JSON body a client would send for row i of a game (earlier rows as history)."""
    r = game.iloc[i]
    return {
        "pitcher_id": int(r["pitcher"]), "pitcher_throws": r["p_throws"],
        "batter_id": int(r["batter"]), "batter_stands": r["stand"],
        "at_bat_number": int(r["at_bat_number"]), "pitch_number": int(r["pitch_number"]),
        "inning": int(r["inning"]), "inning_half": r["inning_topbot"],
        "balls": int(r["balls"]), "strikes": int(r["strikes"]), "outs": int(r["outs_when_up"]),
        "runner_on_1b": bool(pd.notna(r["on_1b"])), "runner_on_2b": bool(pd.notna(r["on_2b"])),
        "runner_on_3b": bool(pd.notna(r["on_3b"])),
        "pitching_team_score": int(r["fld_score"]), "batting_team_score": int(r["bat_score"]),
        "previous_pitches": [
            {"at_bat_number": int(p.at_bat_number), "pitch_number": int(p.pitch_number),
             "batter_id": int(p.batter), "pitch_type": p.target, "result": p.description}
            for p in game.iloc[:i].itertuples()],
    }

def as_array(body: dict) -> np.ndarray:
    got = {p["pitch_type"]: p["probability"] for p in body["probabilities"]}
    return np.array([got[c] for c in CLASSES])

def sample_body(trained):
    raw = trained[0]
    game = raw[raw["game_pk"] == raw["game_pk"].iloc[0]].reset_index(drop=True)
    return api_request(game, 12)

def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_trained_on_seasons"] == [2023]

def test_predict_returns_ten_probabilities_that_sum_to_one(client, trained):
    r = client.post("/predict", json=sample_body(trained))
    assert r.status_code == 200
    body = r.json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert len(probs) == 10 and abs(sum(probs) - 1) < 1e-9
    assert probs == sorted(probs, reverse=True)
    assert body["top_pick"] == body["probabilities"][0]["pitch_type"]

@pytest.mark.parametrize("change", [
    {"balls": 4}, {"strikes": 3}, {"outs": 3}, {"inning": 0}, {"pitcher_throws": "X"},
    {"batter_stands": "S"}, {"inning_half": "Middle"}, {"pitching_team_score": -1},
    {"pitch_number": 0}, {"pitcher_id": "abc"},
])
def test_impossible_input_is_rejected_with_422(client, trained, change):
    r = client.post("/predict", json={**sample_body(trained), **change})
    assert r.status_code == 422

def test_unknown_pitch_type_in_history_is_rejected(client, trained):
    body = sample_body(trained)
    body["previous_pitches"][0]["pitch_type"] = "ZZ"
    assert client.post("/predict", json=body).status_code == 422

def test_missing_field_is_rejected(client, trained):
    body = sample_body(trained)
    del body["balls"]
    assert client.post("/predict", json=body).status_code == 422

def test_pitcher_the_model_never_saw_is_rejected(client, trained):
    r = client.post("/predict", json={**sample_body(trained), "pitcher_id": 99999})
    assert r.status_code == 422 and "not one of the pitchers" in r.json()["detail"]

def test_history_in_the_wrong_order_is_rejected(client, trained):
    body = sample_body(trained)
    body["previous_pitches"] = list(reversed(body["previous_pitches"]))
    assert client.post("/predict", json=body).status_code == 422

def test_api_gives_exactly_the_offline_models_probabilities_for_every_pitch(trained):
    """Training/serving skew test: same model, same data, offline vs through the API."""
    raw, path = trained
    model = load_model(path)
    with TestClient(create_app(model_dir=path)) as client:
        for pk in raw["game_pk"].unique()[:2]:
            game = raw[raw["game_pk"] == pk].reset_index(drop=True)
            offline = model.predict_proba(build_features(game))
            for i in range(len(game)):
                body = client.post("/predict", json=api_request(game, i)).json()
                np.testing.assert_allclose(as_array(body), offline[i], rtol=1e-9, atol=1e-12)

@pytest.mark.skipif(not (REAL_MODEL.exists() and REAL_DATA.exists()),
                    reason="saved model or clean data not available")
def test_real_saved_model_matches_offline_on_real_games():
    model = load_model(REAL_MODEL)
    df = pd.read_parquet(REAL_DATA)
    pid = [p for p in model.enc_.pitchers_ if p in set(df["pitcher"])][0]
    d = df[df["pitcher"] == pid]
    game = d[d["game_pk"] == d["game_pk"].iloc[-1]].reset_index(drop=True)
    offline = model.predict_proba(build_features(game))
    with TestClient(create_app(model_dir=str(REAL_MODEL))) as client:
        for i in range(0, len(game), 5):
            body = client.post("/predict", json=api_request(game, i)).json()
            np.testing.assert_allclose(as_array(body), offline[i], rtol=1e-9, atol=1e-12)

# ---------------- endpoints that read the database ----------------
@pytest.fixture
def db_client(trained, engine):
    clean = make_clean(seasons=(2023,), pitchers=(11, 22), games=2)
    load_clean(engine, clean)
    with TestClient(create_app(model_dir=trained[1], engine=engine)) as c:
        yield c, clean

@needs_db
def test_pitchers_lists_database_pitchers_and_model_coverage(db_client):
    client, _ = db_client
    rows = client.get("/pitchers").json()
    assert {r["pitcher_id"] for r in rows} == {11, 22}
    assert all(r["in_model"] for r in rows) and all(r["throws"] in ("L", "R") for r in rows)

@needs_db
def test_game_returns_its_pitches_in_order(db_client):
    client, clean = db_client
    pk = int(clean["game_pk"].iloc[0])
    body = client.get(f"/games/{pk}").json()
    assert body["game_pk"] == pk and body["season"] == 2023
    assert len(body["pitches"]) == int((clean["game_pk"] == pk).sum())
    order = [(p["at_bat_number"], p["pitch_number"]) for p in body["pitches"]]
    assert order == sorted(order)

@needs_db
def test_unknown_game_is_404(db_client):
    assert db_client[0].get("/games/123456789").status_code == 404

@needs_db
def test_health_reports_database_up(db_client):
    assert db_client[0].get("/health").json()["database"] is True

def test_database_endpoints_say_503_when_no_database_is_configured(trained, monkeypatch):
    monkeypatch.setattr("api.main.find_database_url", lambda: None)
    with TestClient(create_app(model_dir=trained[1])) as c:
        assert c.get("/pitchers").status_code == 503
        assert c.get("/games/1").status_code == 503
        assert c.get("/health").json()["database"] is False
        assert c.post("/predict", json=sample_body(trained)).status_code == 200

@needs_db
def test_starting_against_an_empty_database_creates_the_tables(trained):
    import uuid
    name = f"test_{uuid.uuid4().hex[:10]}"
    admin = make_engine()
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {name}"))
    empty = make_engine(schema=name)          # a brand-new database: no tables yet
    try:
        with TestClient(create_app(model_dir=trained[1], engine=empty)) as client:
            assert client.get("/pitchers").json() == []
    finally:
        empty.dispose()
        with admin.begin() as c:
            c.execute(text(f"DROP SCHEMA {name} CASCADE"))
        admin.dispose()

def test_the_api_still_starts_and_predicts_when_the_database_is_unreachable(trained):
    dead = make_engine("postgresql+psycopg://nobody:nothing@127.0.0.1:1/none")
    with TestClient(create_app(model_dir=trained[1], engine=dead)) as client:
        assert client.get("/health").json()["database"] is False
        assert client.post("/predict", json=sample_body(trained)).status_code == 200