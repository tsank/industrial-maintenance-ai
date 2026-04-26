"""
memory_storage.py — Persists the completed episode to memory_episodes and
appends it to session_history on state. Runs as the final node in the graph.
Reuses state.query_embedding computed by memory_retrieval_node — no redundant
embedding call.
"""

from __future__ import annotations

from datetime import datetime

from compressor_longtermrag.memory_store import write_episode
from compressor_longtermrag.state import MemoryEntry


def memory_storage_node(state) -> dict:
    #only persist episodes that meet the quality threshold
    if state.score >= state.min_score_threshold:
        entry = MemoryEntry(
            session_id=state.session_id,
            query=state.query,
            answer=state.answer,
            score=state.score,
            store_used=state.fallback_store if state.fallback_store else state.query_type,
            created_at=datetime.utcnow(),
        )
        write_episode(entry, state.query_embedding)
        updated_history = state.session_history + [entry]
    else:
        updated_history = state.session_history
        
    return {"session_history": updated_history}