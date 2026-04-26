-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Episode table — one row per completed query-answer cycle
CREATE TABLE IF NOT EXISTS memory_episodes (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    query       TEXT NOT NULL,
    answer      TEXT NOT NULL,
    score       FLOAT,
    store_used  TEXT,
    embedding   vector(1536),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);