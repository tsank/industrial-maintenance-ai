"""
init_db.py — One-time setup: enables pgvector and creates memory_episodes table.

Run once before starting the agent:
    cd v7_observability
    python scripts/init_db.py
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

SCHEMA_PATH = Path(__file__).parent.parent / "data" / "seed" / "init_memory_schema.sql"


def main() -> None:
    conn = psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", 5432)),
        dbname=os.environ["PG_DB"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )
    conn.autocommit = True
    cur = conn.cursor()

    sql = SCHEMA_PATH.read_text()
    cur.execute(sql)

    cur.execute("SELECT COUNT(*) FROM memory_episodes;")
    count = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"Schema ready. memory_episodes row count: {count}")


if __name__ == "__main__":
    main()
