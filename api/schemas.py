from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

PitchCode = Literal["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV"]
Hand = Literal["L", "R"]

class PreviousPitch(BaseModel):
    """A pitch this pitcher already threw earlier in this game."""
    at_bat_number: int = Field(ge=1)
    pitch_number: int = Field(ge=1, description="the pitch's number within its at-bat")
    batter_id: int
    pitch_type: PitchCode
    result: str = Field(description="what happened, e.g. ball, called_strike, swinging_strike, "
                                    "foul, hit_into_play")

class PredictRequest(BaseModel):
    """The situation just before the next pitch."""
    pitcher_id: int
    pitcher_throws: Hand
    batter_id: int
    batter_stands: Hand
    at_bat_number: int = Field(ge=1)
    pitch_number: int = Field(ge=1, description="number of this pitch within the at-bat")
    inning: int = Field(ge=1)
    inning_half: Literal["Top", "Bot"]
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    outs: int = Field(ge=0, le=2)
    runner_on_1b: bool = False
    runner_on_2b: bool = False
    runner_on_3b: bool = False
    pitching_team_score: int = Field(ge=0)
    batting_team_score: int = Field(ge=0)
    previous_pitches: list[PreviousPitch] = Field(
        default_factory=list, max_length=500,
        description="this pitcher's earlier pitches in this game, oldest first")

    model_config = ConfigDict(json_schema_extra={"example": {
        "pitcher_id": 592332, "pitcher_throws": "R", "batter_id": 660271,
        "batter_stands": "L", "at_bat_number": 2, "pitch_number": 2, "inning": 1,
        "inning_half": "Bot", "balls": 0, "strikes": 1, "outs": 0,
        "runner_on_1b": False, "runner_on_2b": False, "runner_on_3b": False,
        "pitching_team_score": 0, "batting_team_score": 0,
        "previous_pitches": [
            {"at_bat_number": 1, "pitch_number": 1, "batter_id": 605141,
             "pitch_type": "FF", "result": "called_strike"},
            {"at_bat_number": 2, "pitch_number": 1, "batter_id": 660271,
             "pitch_type": "FF", "result": "called_strike"}]}})

class PitchProbability(BaseModel):
    pitch_type: PitchCode
    name: str
    probability: float

class PredictResponse(BaseModel):
    pitcher_id: int
    top_pick: PitchCode
    probabilities: list[PitchProbability] = Field(description="all 10 pitch types, most likely first")
    model_trained_on_seasons: list[int]

class PitcherOut(BaseModel):
    pitcher_id: int
    name: str
    throws: Hand
    in_model: bool = Field(description="True if /predict can make forecasts for this pitcher")

class GamePitch(BaseModel):
    at_bat_number: int
    pitch_number: int
    pitcher_id: int
    batter_id: int
    batter_stands: Hand
    inning: int
    inning_half: str
    balls: int
    strikes: int
    outs: int
    pitch_class: PitchCode
    result: str

class GameOut(BaseModel):
    game_pk: int
    game_date: str
    season: int
    home_team: str
    away_team: str
    pitches: list[GamePitch]