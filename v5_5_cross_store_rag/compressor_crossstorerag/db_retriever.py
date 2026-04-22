"""
db_retriever.py — Thin wrapper around psycopg2 for direct SQL queries.

Used in tests to verify that seed data loaded correctly and that
Text-to-SQL results can be validated against raw queries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def get_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", 5432)),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
