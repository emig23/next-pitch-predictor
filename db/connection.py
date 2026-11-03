import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

def find_database_url(env_file: str = ".env") -> str | None:
    """DATABASE_URL from the environment, else from a .env file in the current folder."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    path = Path(env_file)
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "DATABASE_URL" and value.strip():
                return value.strip()
    return None

def get_database_url() -> str:
    url = find_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in "
                           "(or set the DATABASE_URL environment variable).")
    return url

def make_engine(url: str | None = None, schema: str | None = None) -> Engine:
    connect_args = {"connect_timeout": 5}
    if schema:
        connect_args["options"] = f"-csearch_path={schema}"
    return create_engine(url or get_database_url(), connect_args=connect_args)

def apply_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_FILE.read_text())