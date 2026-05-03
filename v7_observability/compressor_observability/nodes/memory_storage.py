"""
memory_storage.py — Persists the completed episode to memory_episodes and
appends it to session_history on state. Runs as the final node in the graph.
Reuses state.query_embedding computed by memory_retrieval_node — no redundant
embedding call.
"""

from __future__ import annotations

from datetime import datetime, timezone

from compressor_observability.memory_store import write_episode
from compressor_observability.state import MemoryEntry
from compressor_observability.observability import traced_node


@traced_node("node.memory_storage")
def memory_storage_node(state, span) -> dict:
    #only persist episodes that meet the quality threshold
    persisted = state.score >= state.min_score_threshold
    if persisted:
        entry = MemoryEntry(
            session_id=state.session_id,
            query=state.query,
            answer=state.answer,
            score=state.score,
            store_used=state.fallback_store if state.fallback_store else state.query_type,
            created_at=datetime.now(timezone.utc),
        )
        write_episode(entry, state.query_embedding)
        updated_history = state.session_history + [entry]
    else:
        updated_history = state.session_history

    span.set_attribute("final_score", state.score)
    span.set_attribute("store_used", state.fallback_store if state.fallback_store else state.query_type)
    span.set_attribute("persisted", persisted)
        
    return {"session_history": updated_history}