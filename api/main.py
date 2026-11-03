"""
GET  /health          is the service up, is the model loaded
GET  /pitchers        pitchers in the database and whether the model can forecast them
GET  /games/{game_pk} one game's pitches, in order
POST /predict         probabilities for the next pitch given a game situation
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import text
from api.predict import PITCH_NAMES, predict_probabilities
from api.schemas import (GameOut, GamePitch, PitcherOut, PitchProbability, PredictRequest,
                         PredictResponse)
from db.connection import apply_schema, find_database_url, make_engine
from src.modeling.persistence import load_model

DEFAULT_MODEL_DIR = "models/global_xgb"
log = logging.getLogger("api")

def create_app(model_dir: str | None = None, engine=None) -> FastAPI:
    """`engine` can be passed in (the tests do); otherwise it is built from DATABASE_URL
    if one is set. Without a database the prediction endpoint still works."""
    model_dir = model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_model(model_dir)          # loaded once, not per request
        meta = json.loads((Path(model_dir) / "meta.json").read_text())
        app.state.trained_on = meta.get("extra", {}).get("trained_on_seasons", [])
        app.state.engine = engine or (make_engine() if find_database_url() else None)
        if app.state.engine is not None:
            try:        # create the tables if the database is brand new (safe to repeat)
                apply_schema(app.state.engine)
            except Exception as exc:                       # noqa: BLE001
                log.warning("database not ready at startup (%s); predictions still work", exc)
        yield
        if app.state.engine is not None and engine is None:
            app.state.engine.dispose()

    app = FastAPI(title="Next Pitch Predictor API", version="1.0.0", lifespan=lifespan,
                  description="Forecasts the probability of each pitch type before the "
                              "pitch is thrown.")

    def db(request: Request):
        eng = request.app.state.engine
        if eng is None:
            raise HTTPException(503, "No database is configured (set DATABASE_URL).")
        return eng

    @app.get("/health")
    def health(request: Request):
        eng = request.app.state.engine
        database = False
        if eng is not None:
            try:
                with eng.connect() as conn:
                    conn.execute(text("SELECT 1"))
                database = True
            except Exception:                              # noqa: BLE001
                database = False
        return {"status": "ok", "model_loaded": True, "database": database,
                "model_trained_on_seasons": request.app.state.trained_on}

    @app.get("/pitchers", response_model=list[PitcherOut])
    def pitchers(request: Request):
        known = set(request.app.state.model.enc_.pitchers_)
        with db(request).connect() as conn:
            rows = conn.execute(text(
                "SELECT pitcher_id, name, throws FROM pitchers ORDER BY name")).all()
        return [PitcherOut(pitcher_id=r[0], name=r[1], throws=r[2], in_model=r[0] in known)
                for r in rows]

    @app.get("/games/{game_pk}", response_model=GameOut)
    def game(game_pk: int, request: Request):
        eng = db(request)
        with eng.connect() as conn:
            g = conn.execute(text(
                "SELECT game_pk, game_date::text, season, home_team, away_team "
                "FROM games WHERE game_pk = :g"), {"g": game_pk}).first()
            if g is None:
                raise HTTPException(404, f"No game with game_pk {game_pk}")
            rows = conn.execute(text(
                "SELECT at_bat_number, pitch_number, pitcher_id, batter_id, stand, inning, "
                "inning_topbot, balls, strikes, outs_when_up, pitch_class, description "
                "FROM pitches WHERE game_pk = :g ORDER BY at_bat_number, pitch_number"),
                {"g": game_pk}).all()
        pitches = [GamePitch(at_bat_number=r[0], pitch_number=r[1], pitcher_id=r[2],
                             batter_id=r[3], batter_stands=r[4], inning=r[5],
                             inning_half=r[6], balls=r[7], strikes=r[8], outs=r[9],
                             pitch_class=r[10], result=r[11]) for r in rows]
        return GameOut(game_pk=g[0], game_date=g[1], season=g[2], home_team=g[3],
                       away_team=g[4], pitches=pitches)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request):
        try:
            probs = predict_probabilities(request.app.state.model, req)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return PredictResponse(
            pitcher_id=req.pitcher_id, top_pick=ranked[0][0],
            probabilities=[PitchProbability(pitch_type=c, name=PITCH_NAMES[c], probability=p)
                           for c, p in ranked],
            model_trained_on_seasons=request.app.state.trained_on)

    return app

app = create_app()