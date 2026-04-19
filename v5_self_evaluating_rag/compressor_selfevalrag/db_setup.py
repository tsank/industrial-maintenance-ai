"""
db_setup.py — Create PostgreSQL tables and load seed CSV data.

Run once:
    python -m compressor_multistorerag.db_setup

Tables created:
    operational_parameters   single-value parameters (pressure, oil, etc.)
    pressure_settings        per-model pressure setpoints
    service_plans            A/B/C/D plans with intervals
    protection_thresholds    shutdown and warning levels
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

_SEED_DIR = Path(__file__).parent.parent / "data" / "seed"

DDL = """
CREATE TABLE IF NOT EXISTS operational_parameters (
    parameter  TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    unit       TEXT
);

CREATE TABLE IF NOT EXISTS pressure_settings (
    ga_model         TEXT    NOT NULL,
    icd_model        TEXT    NOT NULL,
    dewpoint_variant TEXT    NOT NULL,
    frequency_hz     INTEGER NOT NULL,
    unload_pressure_bar FLOAT,
    load_pressure_bar   FLOAT,
    PRIMARY KEY (ga_model, icd_model, dewpoint_variant, frequency_hz)
);

CREATE TABLE IF NOT EXISTS service_plans (
    plan             CHAR(1) PRIMARY KEY,
    interval_hours   INTEGER,
    interval_months  INTEGER,
    description      TEXT
);

CREATE TABLE IF NOT EXISTS protection_thresholds (
    parameter TEXT  NOT NULL,
    level     TEXT  NOT NULL,   -- 'shutdown' or 'warning'
    value     FLOAT NOT NULL,
    unit      TEXT,
    PRIMARY KEY (parameter, level)
);
"""


def _get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", 5432)),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )


def create_tables(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    print("Tables created (or already exist).")


def _load_csv(conn: psycopg2.extensions.connection, table: str, csv_path: Path) -> None:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    col_str = ", ".join(cols)
    sql = (
        f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )
    with conn.cursor() as cur:
        for row in rows:
            # convert empty strings to None
            values = [v if v != "" else None for v in row.values()]
            cur.execute(sql, values)
    conn.commit()
    print(f"  Loaded {len(rows)} rows into {table}.")


def load_seed_data(conn: psycopg2.extensions.connection) -> None:
    mapping = {
        "operational_parameters": _SEED_DIR / "operational_parameters.csv",
        "pressure_settings": _SEED_DIR / "pressure_settings.csv",
        "service_plans": _SEED_DIR / "service_plans.csv",
        "protection_thresholds": _SEED_DIR / "protection_thresholds.csv",
    }
    print("Loading seed data...")
    for table, path in mapping.items():
        if path.exists():
            _load_csv(conn, table, path)
        else:
            print(f"  WARNING: seed file not found: {path}")


def main() -> None:
    conn = _get_conn()
    try:
        create_tables(conn)
        load_seed_data(conn)
        print("Setup complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
