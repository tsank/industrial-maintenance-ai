"""
test_memory_retrieval.py — Verifies that retrieved_memories is populated
when semantically similar past episodes exist in memory_episodes.

Seeds the database with a known episode, then queries with a related
question and asserts retrieved_memories is non-empty.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from compressor_observability.memory_store import write_episode, search_similar_episodes
from compressor_observability.state import MemoryEntry
from compressor_observability.nodes.clients import embeddings
from compressor_observability.agent import run


def _seed_episode(query: str, answer: str) -> None:
    entry = MemoryEntry(
        session_id=f"seed_{uuid.uuid4().hex}",
        query=query,
        answer=answer,
        score=0.9,
        store_used="spec",
        created_at=datetime.utcnow(),
    )
    embedding = embeddings.embed_query(query)
    write_episode(entry, embedding)


def test_retrieved_memories_populated():
    # Seed a known episode
    _seed_episode(
        query="What is the maximum working pressure of the GA5?",
        answer="The maximum working pressure of the GA5 is 8.5 bar.",
    )

    # Run with a semantically related query
    state = run(
        query="What is the rated operating pressure for the GA5 compressor?",
        session_id=f"test_retrieval_{uuid.uuid4().hex}",
    )

    # At least one past episode should have been retrieved
    assert state.memory_injected is True
    assert len(state.retrieved_memories) >= 1