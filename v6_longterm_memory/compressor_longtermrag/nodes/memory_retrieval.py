"""
memory_retrieval.py — Fetches top-k semantically similar past episodes from
memory_episodes and writes them into retrieved_memories on state.
Computes and stores query_embedding on state to avoid re-embedding in
memory_storage_node.
Runs as the first node in the graph, before classify.
"""

from __future__ import annotations

import os

from compressor_longtermrag.nodes.clients import embeddings
from compressor_longtermrag.memory_store import search_similar_episodes

_MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", 3))


def memory_retrieval_node(state) -> dict:
    query_embedding = embeddings.embed_query(state.query)
    memories = search_similar_episodes(query_embedding, top_k=_MEMORY_TOP_K)
    return {
        "retrieved_memories": memories,
        "memory_injected": True,
        "query_embedding": query_embedding,
    }