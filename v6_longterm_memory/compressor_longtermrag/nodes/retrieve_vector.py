"""
retrieve_vector.py — ChromaDB vector retrieval node. Carried over from v5.5 unchanged.
"""

from __future__ import annotations

from compressor_longtermrag.nodes.clients import embeddings, collection


def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"


def vector_retrieve_node(state) -> dict:
    query_vec = embeddings.embed_query(state.query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=4,
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs) if docs else "No relevant passages found."
    accumulated = _append_context(state.accumulated_context, context)
    return {"vector_context": context, "accumulated_context": accumulated}