import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy import MetaData

PITCH_COLUMNS = [
    "game_pk", "at_bat_number", "pitch_number", "pitcher_id", "batter_id", "stand", "inning",
    "inning_topbot", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "home_score", "away_score", "bat_score", "fld_score", "pitch_type", "pitch_class",
    "description", "type", "n_thruorder_pitcher",
]

# clean-data column -> database column
RENAME = {"pitcher": "pitcher_id", "batter": "batter_id", "target": "pitch_class",
          "player_name": "name", "p_throws": "throws"}
RUNNER_COLUMNS = ["on_1b", "on_2b", "on_3b"]
SORT_KEYS = ["pitcher", "game_pk", "at_bat_number", "pitch_number"]
CHUNK = 5000

def _one_per_key(df: pd.DataFrame, key: str, cols: list, what: str) -> pd.DataFrame:
    out = df[[key] + cols].drop_duplicates()
    if out[key].duplicated().any():
        bad = out.loc[out[key].duplicated(keep=False), key].unique()[:5].tolist()
        raise ValueError(f"to_tables: {what} has more than one value for {key} {bad}")
    return out.reset_index(drop=True)

def to_tables(clean: pd.DataFrame):
    df = clean.rename(columns=RENAME)
    if "n_thruorder_pitcher" not in df.columns:
        df["n_thruorder_pitcher"] = pd.NA
    game_date = pd.to_datetime(df["game_date"])
    if "season" in df.columns and not (game_date.dt.year == df["season"]).all():
        raise ValueError("to_tables: season does not match game_date")
    df = df.assign(game_date=game_date.dt.date, season=game_date.dt.year)

    pitchers = _one_per_key(df, "pitcher_id", ["name", "throws"], "pitchers")
    games = _one_per_key(df, "game_pk", ["game_date", "season", "home_team", "away_team"],
                         "games")
    pitches = df[PITCH_COLUMNS].copy()
    for c in RUNNER_COLUMNS + ["n_thruorder_pitcher"]:   # nullable integers, not floats
        pitches[c] = pitches[c].astype("Int64")
    return pitchers, games, pitches.reset_index(drop=True)

def _records(df: pd.DataFrame) -> list:
    clean = df.astype(object).where(df.notna(), None)
    return clean.to_dict("records")

def _insert_new(engine: Engine, table_name: str, df: pd.DataFrame) -> int:
    meta = MetaData()
    meta.reflect(engine, only=[table_name])
    table = meta.tables[table_name]
    pk_column = list(table.primary_key.columns)[0]
    stmt = insert(table).on_conflict_do_nothing().returning(pk_column)
    records, added = _records(df), 0
    with engine.begin() as conn:
        for i in range(0, len(records), CHUNK):
            added += len(conn.execute(stmt, records[i:i + CHUNK]).all())
    return added

def load_clean(engine: Engine, clean: pd.DataFrame) -> dict:
    """Insert whatever is new. Returns how many rows were added to each table."""
    pitchers, games, pitches = to_tables(clean)
    return {
        "pitchers": _insert_new(engine, "pitchers", pitchers),   # parents first, because
        "games": _insert_new(engine, "games", games),            # pitches point to them
        "pitches": _insert_new(engine, "pitches", pitches),
    }

def count_rows(engine: Engine) -> dict:
    with engine.connect() as conn:
        return {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
                for t in ("pitchers", "games", "pitches")}

def read_clean(engine: Engine) -> pd.DataFrame:
    """All pitches in the layout the model code uses, sorted the way build_features needs."""
    sql = """
        SELECT p.game_pk, g.game_date::text AS game_date, p.pitcher_id AS pitcher,
               pi.name AS player_name, p.batter_id AS batter, p.stand, pi.throws AS p_throws,
               g.home_team, g.away_team, p.inning, p.inning_topbot, p.balls, p.strikes,
               p.outs_when_up, p.on_1b, p.on_2b, p.on_3b, p.bat_score, p.fld_score,
               p.home_score, p.away_score, p.at_bat_number, p.pitch_number, p.pitch_type,
               p.description, p.type, p.n_thruorder_pitcher, g.season,
               p.pitch_class AS target
        FROM pitches p
        JOIN games g USING (game_pk)
        JOIN pitchers pi ON pi.pitcher_id = p.pitcher_id
        ORDER BY p.pitcher_id, p.game_pk, p.at_bat_number, p.pitch_number"""
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)