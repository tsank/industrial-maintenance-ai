"""
retrieve_vector.py — ChromaDB vector retrieval node. Changes with respect to v6 are as follows:
1) Change the import statement to import from `compressor_observability` module
2) Update definition of `vector_retrieve_node()` to add `span`
3) Set the attributes for the span - doc length (returned by ChromaDB) and context length
"""

from __future__ import annotations

from compressor_observability.nodes.clients import embeddings, collection
from compressor_observability.observability import traced_node

def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"

@traced_node("node.retrieve_vector")
def vector_retrieve_node(state, span) -> dict:
    query_vec = embeddings.embed_query(state.query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=4,
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs) if docs else "No relevant passages found."
    accumulated = _append_context(state.accumulated_context, context)
    span.set_attribute("chunks_returned", len(docs))     # Did ChromaDB find anything?
    span.set_attribute("context_length", len(context))   # Is the content meaningful or empty?
    return {"vector_context": context, "accumulated_context": accumulated}