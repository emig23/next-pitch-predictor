import re
import uuid
from pathlib import Path
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from db.connection import apply_schema, find_database_url, make_engine
from db.load import _insert_new, count_rows, load_clean, read_clean, to_tables
from tests.test_final import make_raw

def make_clean(**kw):
    """make_raw() gives the cleaned-data columns we need; add the few it lacks."""
    df = make_raw(**kw)
    df["game_date"] = df["season"].astype(str) + "-06-01"
    df["home_team"], df["away_team"] = "AAA", "BBB"
    df["home_score"], df["away_score"] = 0, 0
    df["pitch_type"], df["type"] = df["target"], "B"
    df["n_thruorder_pitcher"] = 1
    return df

# ---------------- no database needed ----------------

def test_to_tables_splits_without_losing_or_repeating_rows():
    clean = make_clean(seasons=(2023, 2024), pitchers=(11, 22), games=3)
    pitchers, games, pitches = to_tables(clean)
    assert len(pitches) == len(clean)
    assert pitchers["pitcher_id"].is_unique and len(pitchers) == 2
    assert games["game_pk"].is_unique and len(games) == clean["game_pk"].nunique()
    assert set(games["season"]) == {2023, 2024}
    assert not pitches.duplicated(["game_pk", "at_bat_number", "pitch_number"]).any()

def test_to_tables_keeps_runners_as_integers_or_missing():
    clean = make_clean(seasons=(2023,), pitchers=(11,), games=2)
    _, _, pitches = to_tables(clean)
    assert str(pitches["on_1b"].dtype) == "Int64"
    assert pitches["on_1b"].isna().any() and (pitches["on_1b"].dropna() == 999).all()

def test_to_tables_rejects_a_pitcher_with_two_names():
    clean = make_clean(seasons=(2023,), pitchers=(11,), games=1)
    clean.loc[clean.index[0], "player_name"] = "Someone Else"
    with pytest.raises(ValueError, match="more than one value"):
        to_tables(clean)

def test_to_tables_rejects_season_that_disagrees_with_date():
    clean = make_clean(seasons=(2023,), pitchers=(11,), games=1)
    clean["season"] = 2024
    with pytest.raises(ValueError, match="season"):
        to_tables(clean)

# ---------------- needs a real Postgres ----------------

needs_db = pytest.mark.skipif(find_database_url() is None,
                              reason="DATABASE_URL not set (env var or .env)")

@pytest.fixture
def engine():
    name = f"test_{uuid.uuid4().hex[:10]}"
    admin = make_engine()
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {name}"))
    eng = make_engine(schema=name)
    apply_schema(eng)
    yield eng
    eng.dispose()
    with admin.begin() as c:
        c.execute(text(f"DROP SCHEMA {name} CASCADE"))
    admin.dispose()

@needs_db
def test_schema_can_be_applied_twice(engine):
    apply_schema(engine)

@needs_db
def test_row_counts_match_the_source_data(engine):
    clean = make_clean(seasons=(2023, 2024), pitchers=(11, 22), games=3)
    added = load_clean(engine, clean)
    have = count_rows(engine)
    assert have["pitches"] == len(clean) == added["pitches"]
    assert have["pitchers"] == 2 and have["games"] == clean["game_pk"].nunique()

@needs_db
def test_loading_twice_never_duplicates(engine):
    clean = make_clean(seasons=(2023,), pitchers=(11, 22), games=3)
    load_clean(engine, clean)
    before = count_rows(engine)
    again = load_clean(engine, clean)
    assert again == {"pitchers": 0, "games": 0, "pitches": 0}
    assert count_rows(engine) == before

@needs_db
def test_loading_fills_in_only_the_missing_rows(engine):
    clean = make_clean(seasons=(2023,), pitchers=(11,), games=3)
    load_clean(engine, clean.iloc[:100])
    added = load_clean(engine, clean)
    assert added["pitches"] == len(clean) - 100
    assert count_rows(engine)["pitches"] == len(clean)

@needs_db
def test_data_read_back_equals_data_put_in(engine):
    clean = make_clean(seasons=(2023, 2025), pitchers=(11, 22), games=2)
    load_clean(engine, clean)
    back = read_clean(engine)
    cols = ["game_pk", "pitcher", "batter", "stand", "p_throws", "inning", "inning_topbot",
            "balls", "strikes", "outs_when_up", "bat_score", "fld_score", "at_bat_number",
            "pitch_number", "description", "season", "target", "game_date", "player_name"]
    want = clean.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"]
                             ).reset_index(drop=True)
    pd.testing.assert_frame_equal(back[cols], want[cols], check_dtype=False)
    for c in ("on_1b", "on_2b", "on_3b"):          # runner present/absent preserved
        assert (back[c].notna().to_numpy() == want[c].notna().to_numpy()).all()

@needs_db
def test_database_rejects_impossible_rows(engine):
    clean = make_clean(seasons=(2023,), pitchers=(11,), games=1)
    load_clean(engine, clean.iloc[:5])
    row = clean.iloc[:1].copy()
    row["pitch_number"] = 99
    for column, value in [("balls", 4), ("strikes", 3), ("target", "ZZ")]:
        bad = row.copy()
        bad[column] = value
        with pytest.raises(IntegrityError):
            load_clean(engine, bad)
    orphan = row.copy()                             # pitcher not in the pitchers table
    orphan["pitcher"] = 12345
    with pytest.raises(IntegrityError):
        _insert_new(engine, "pitches", to_tables(orphan)[2])

@needs_db
def test_every_example_query_is_valid_sql(engine):
    sql = Path("db/queries.sql").read_text()
    parts = [p for p in re.split(r"-- name: \w+", sql)[1:]]
    assert len(parts) >= 4
    with engine.connect() as conn:
        for part in parts:
            conn.execute(text("EXPLAIN " + part.strip().rstrip(";")))

def test_database_url_can_come_from_an_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    f = tmp_path / ".env"
    f.write_text("POSTGRES_USER=x\nDATABASE_URL=postgresql+psycopg://u:p@h:1/d\n")
    assert find_database_url(str(f)) == "postgresql+psycopg://u:p@h:1/d"
    assert find_database_url(str(tmp_path / "missing.env")) is None
    monkeypatch.setenv("DATABASE_URL", "from-env")
    assert find_database_url(str(f)) == "from-env"