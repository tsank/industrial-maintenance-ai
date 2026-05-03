"""
memory_retrieval.py — Fetches top-k semantically similar past episodes from
memory_episodes and writes them into retrieved_memories on state.
Computes and stores query_embedding on state to avoid re-embedding in
memory_storage_node.
Runs as the first node in the graph, before classify.
"""

from __future__ import annotations

import os

from compressor_observability.nodes.clients import embeddings
from compressor_observability.memory_store import search_similar_episodes

_MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", 3))


from compressor_observability.observability import traced_node

@traced_node("node.memory_retrieval")
def memory_retrieval_node(state, span) -> dict:
    query_embedding = embeddings.embed_query(state.query)
    memories = search_similar_episodes(query_embedding, top_k=_MEMORY_TOP_K)
    span.set_attribute("episodes_retrieved", len(memories))
    return {
        "retrieved_memories": memories,
        "memory_injected": True,
        "query_embedding": query_embedding,
    }