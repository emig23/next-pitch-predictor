from src.dataset.features import CLASSES
from src.dataset.request import features_for_request
from api.schemas import PredictRequest

PITCH_NAMES = {
    "FF": "Four-seam fastball", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle curve", "CH": "Changeup",
    "FS": "Splitter", "SV": "Slurve",
}

def to_feature_inputs(req: PredictRequest):
    current = dict(
        pitcher=req.pitcher_id, batter=req.batter_id, at_bat_number=req.at_bat_number,
        pitch_number=req.pitch_number, balls=req.balls, strikes=req.strikes,
        outs_when_up=req.outs, inning=req.inning, inning_topbot=req.inning_half,
        on_1b=req.runner_on_1b, on_2b=req.runner_on_2b, on_3b=req.runner_on_3b,
        bat_score=req.batting_team_score, fld_score=req.pitching_team_score,
        stand=req.batter_stands, p_throws=req.pitcher_throws)
    history = [dict(at_bat_number=p.at_bat_number, pitch_number=p.pitch_number,
                    batter=p.batter_id, pitch_type=p.pitch_type, description=p.result)
               for p in req.previous_pitches]
    return current, history

def predict_probabilities(model, req: PredictRequest) -> dict:
    if model.enc_.use_pitcher_id and req.pitcher_id not in model.enc_.pitchers_:
        raise ValueError(f"pitcher_id {req.pitcher_id} is not one of the pitchers this "
                         f"model was trained on")
    current, history = to_feature_inputs(req)
    feats = features_for_request(current, history)
    proba = model.predict_proba(feats)[0]
    return {c: float(p) for c, p in zip(CLASSES, proba)}