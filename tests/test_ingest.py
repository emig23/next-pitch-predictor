from datetime import date
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from db.load import count_rows, load_clean
from ingest.ingest import resolve_dates, run, transform
from src.prep.clean_data import clean, select_columns
from tests.test_db import engine, make_clean, needs_db  # noqa: F401  (engine is a fixture)

def statcast_like(**kw) -> pd.DataFrame:
    """What statcast_pitcher returns: regular-season rows, datetime game_date, extra columns."""
    df = make_clean(**kw).drop(columns=["target", "season"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_type"] = "R"
    df["release_speed"] = 93.4         
    df["game_pk"] += 10_000             
    return df                           

def fake_fetch_from(frame: pd.DataFrame):
    calls = []

    def fetch(pitcher_id, start, end):
        calls.append(pitcher_id)
        return frame[frame["pitcher"] == pitcher_id].copy()
    fetch.calls = calls
    return fetch

# ---------------- transform / cleaning (no database) ----------------
def test_transform_drops_non_pitches_missing_types_and_other_game_types():
    df = statcast_like(seasons=(2024,), pitchers=(11,), games=2)
    df.loc[df.index[0], "pitch_type"] = "PO"        # pickoff, not a pitch
    df.loc[df.index[1], "pitch_type"] = None
    df.loc[df.index[2], "game_type"] = "S"          # spring training
    out = transform([df])
    assert len(out) == len(df) - 3
    assert {"season", "target"} <= set(out.columns)

def test_transform_folds_cs_into_cu_but_keeps_raw_pitch_type():
    df = statcast_like(seasons=(2024,), pitchers=(11,), games=1)
    df.loc[df.index[0], "pitch_type"] = "CS"
    out = transform([df])
    row = out[(out["at_bat_number"] == df.loc[df.index[0], "at_bat_number"])
              & (out["pitch_number"] == df.loc[df.index[0], "pitch_number"])].iloc[0]
    assert row["pitch_type"] == "CS" and row["target"] == "CU"

def test_a_day_with_only_some_pitch_types_is_fine_but_an_unknown_type_is_not():
    df = statcast_like(seasons=(2024,), pitchers=(11,), games=1)
    assert df["pitch_type"].nunique() < 10
    assert len(transform([df])) == len(df)        
    df.loc[df.index[0], "pitch_type"] = "ZZ"
    with pytest.raises(ValueError, match="Unexpected target classes"):
        transform([df])

def test_the_full_rebuild_still_requires_all_ten_classes():
    df = select_columns(statcast_like(seasons=(2024,), pitchers=(11,), games=1))
    with pytest.raises(ValueError, match="missing"):
        clean(df, require_all_classes=True)

def test_empty_input_does_not_crash():
    assert transform([]).empty
    assert transform([None, pd.DataFrame()]).empty
    df = statcast_like(seasons=(2024,), pitchers=(11,), games=1)
    df["game_type"] = "S"
    assert transform([df]).empty

def test_missing_required_column_is_reported():
    df = statcast_like(seasons=(2024,), pitchers=(11,), games=1).drop(columns=["balls"])
    with pytest.raises(ValueError, match="balls"):
        transform([df])

def test_resolve_dates():
    today = date(2026, 4, 10)
    assert resolve_dates(None, None, 3, today) == ("2026-04-08", "2026-04-10")
    assert resolve_dates(None, None, None, today) == ("2026-04-08", "2026-04-10")
    assert resolve_dates("2026-04-01", None, None, today) == ("2026-04-01", "2026-04-10")
    assert resolve_dates("2026-04-01", "2026-04-03", None, today) == ("2026-04-01", "2026-04-03")
    with pytest.raises(ValueError):
        resolve_dates("2026-04-05", "2026-04-01", None, today)
    with pytest.raises(ValueError):
        resolve_dates(None, "2026-04-01", None, today)

# ---------------- with a database ----------------
@pytest.fixture
def seeded(engine):
    """Database that already has the pitchers (as after db.load_initial), plus 2023 data."""
    load_clean(engine, make_clean(seasons=(2023,), pitchers=(11, 22), games=2))
    return engine

@needs_db
def test_ingestion_pulls_for_every_pitcher_in_the_database(seeded):
    fetch = fake_fetch_from(statcast_like(seasons=(2024,), pitchers=(11, 22), games=1))
    run(seeded, "2024-06-01", "2024-06-07", fetch=fetch, pause=0)
    assert sorted(fetch.calls) == [11, 22]

@needs_db
def test_an_offseason_range_exits_cleanly_with_zero_rows(seeded):
    before = count_rows(seeded)
    out = run(seeded, "2026-11-01", "2026-11-05",
              fetch=lambda pid, s, e: pd.DataFrame(), pause=0)
    assert out == {"fetched": 0, "usable": 0, "added": 0}
    assert count_rows(seeded) == before
    run(seeded, "2026-11-01", "2026-11-05", fetch=lambda pid, s, e: None, pause=0)

@needs_db
def test_running_the_same_dates_twice_never_duplicates_and_repairs_deleted_rows(seeded):
    new = statcast_like(seasons=(2024,), pitchers=(11, 22), games=2)
    fetch = fake_fetch_from(new)
    first = run(seeded, "2024-06-01", "2024-06-07", fetch=fetch, pause=0)
    assert first["added"] == len(new)
    total = count_rows(seeded)["pitches"]

    again = run(seeded, "2024-06-01", "2024-06-07", fetch=fetch, pause=0)
    assert again["added"] == 0 and count_rows(seeded)["pitches"] == total

    game = int(new["game_pk"].iloc[0])                   
    gone = int((new["game_pk"] == game).sum())
    with seeded.begin() as conn:
        conn.execute(text("DELETE FROM pitches WHERE game_pk = :g"), {"g": game})
    assert count_rows(seeded)["pitches"] == total - gone

    back = run(seeded, "2024-06-01", "2024-06-07", fetch=fetch, pause=0)
    assert back["added"] == gone and count_rows(seeded)["pitches"] == total
    assert run(seeded, "2024-06-01", "2024-06-07", fetch=fetch, pause=0)["added"] == 0

@needs_db
def test_ingested_data_matches_what_the_training_pipeline_would_produce(seeded):
    new = statcast_like(seasons=(2024,), pitchers=(11,), games=2)
    run(seeded, "2024-06-01", "2024-06-07", fetch=fake_fetch_from(new), pause=0)
    with seeded.connect() as conn:
        n = conn.execute(text("SELECT count(*) FROM pitches p JOIN games g USING (game_pk) "
                              "WHERE g.season = 2024")).scalar_one()
    assert n == len(clean(select_columns(new), require_all_classes=False))