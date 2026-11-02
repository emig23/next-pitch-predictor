import argparse
import logging
import time
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import text
from db.connection import make_engine
from db.load import load_clean
from src.prep.clean_data import clean, select_columns

log = logging.getLogger("ingest")

def resolve_dates(start: str | None, end: str | None, days: int | None,
                  today: date | None = None) -> tuple[str, str]:
    today = today or date.today()
    if start:
        s, e = date.fromisoformat(start), date.fromisoformat(end) if end else today
    else:
        if end:
            raise ValueError("--end needs --start (or use --days)")
        s, e = today - timedelta(days=(days if days is not None else 3) - 1), today
    if s > e:
        raise ValueError(f"start {s} is after end {e}")
    return s.isoformat(), e.isoformat()

def get_pitchers(engine) -> list:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT pitcher_id, name FROM pitchers ORDER BY name")).all()
    return [(int(r[0]), r[1]) for r in rows]

def fetch_pitcher(pitcher_id: int, start: str, end: str, tries: int = 3,
                  wait: float = 5.0) -> pd.DataFrame | None:
    from pybaseball import statcast_pitcher
    for attempt in range(1, tries + 1):
        try:
            return statcast_pitcher(start, end, pitcher_id)
        except Exception as exc:
            if attempt == tries:
                raise
            log.warning("pull for %s failed (%s); retry %d/%d", pitcher_id, exc, attempt, tries)
            time.sleep(wait)


def transform(frames: list) -> pd.DataFrame:
    """Raw Statcast frames -> cleaned pitch table (empty if there is nothing usable)."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["game_type"] == "R"]
    if raw.empty:
        return pd.DataFrame()
    return clean(select_columns(raw), require_all_classes=False)

def run(engine, start: str, end: str, fetch=fetch_pitcher, pause: float = 1.0) -> dict:
    pitchers = get_pitchers(engine)
    if not pitchers:
        raise RuntimeError("The pitchers table is empty. Run `python -m db.load_initial` first.")
    log.info("ingesting %s to %s for %d pitchers", start, end, len(pitchers))

    frames, fetched = [], 0
    for pid, name in pitchers:
        df = fetch(pid, start, end)
        n = 0 if df is None else len(df)
        fetched += n
        log.info("  %-24s %5d rows", name, n)
        frames.append(df)
        time.sleep(pause)

    cleaned = transform(frames)
    if cleaned.empty:
        log.info("0 usable rows for %s to %s; nothing to load", start, end)
        return {"fetched": fetched, "usable": 0, "added": 0}

    added = load_clean(engine, cleaned)["pitches"]
    log.info("fetched %d rows, %d usable, %d new pitches inserted", fetched, len(cleaned), added)
    return {"fetched": fetched, "usable": len(cleaned), "added": added}

def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily Statcast ingestion into Postgres")
    ap.add_argument("--start", help="first date, YYYY-MM-DD")
    ap.add_argument("--end", help="last date, YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, help="instead of dates: the last N days (default 3)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start, end = resolve_dates(args.start, args.end, args.days)
    result = run(make_engine(), start, end)
    print(result)

if __name__ == "__main__":
    main()