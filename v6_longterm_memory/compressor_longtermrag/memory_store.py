"""
memory_store.py — PostgreSQL episode CRUD and pgvector similarity search.

All direct interaction with the memory_episodes table is isolated here.
Nodes call write_episode() and search_similar_episodes() — they never
touch the database connection directly.

Module-level connection is initialised at import time, consistent with
the pattern established across v4, v5, and v5.5.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

from compressor_longtermrag.state import MemoryEntry

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Module-level connection
# ---------------------------------------------------------------------------

_pg_conn = psycopg2.connect(
    host=os.environ["PG_HOST"],
    port=int(os.environ.get("PG_PORT", 5432)),
    dbname=os.environ["PG_DB"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
)
_pg_conn.autocommit = True


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_episode(entry: MemoryEntry, embedding: list[float]) -> None:
    """Insert one completed episode into memory_episodes."""
    cur = _pg_conn.cursor()
    cur.execute(
        """
        INSERT INTO memory_episodes
            (session_id, query, answer, score, store_used, embedding, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            entry.session_id,
            entry.query,
            entry.answer,
            entry.score,
            entry.store_used,
            embedding,
            entry.created_at,
        ),
    )
    cur.close()


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def search_similar_episodes(
    query_embedding: list[float],
    top_k: int = 3,
) -> list[MemoryEntry]:
    """
    Return the top_k most similar past episodes using pgvector cosine distance.
    Returns an empty list if the table has fewer rows than ivfflat requires.
    """
    cur = _pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT session_id, query, answer, score, store_used, created_at
        FROM   memory_episodes
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
        """,
        (query_embedding, top_k),
    )
    rows = cur.fetchall()
    cur.close()

    return [
        MemoryEntry(
            session_id=row["session_id"],
            query=row["query"],
            answer=row["answer"],
            score=row["score"] or 0.0,
            store_used=row["store_used"] or "",
            created_at=row["created_at"],
        )
        for row in rows
    ]
