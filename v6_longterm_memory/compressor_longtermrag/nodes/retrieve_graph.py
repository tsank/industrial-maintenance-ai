"""
retrieve_graph.py — NetworkX graph retrieval node. Carried over from v5.5 unchanged.
"""

from __future__ import annotations

import numpy as np

from compressor_longtermrag.nodes.clients import (
    embeddings, G, node_labels, node_matrix, SEMANTIC_THRESHOLD,
)


def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"


def _graph_search(query: str) -> str:
    query_lower = query.lower()

    matched_substring = {
        n for n in G.nodes
        if len(str(n)) > 4 and str(n).lower() in query_lower
    }

    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(node_matrix, query_vec) / (
        np.linalg.norm(node_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    semantic_indices = np.where(similarities >= SEMANTIC_THRESHOLD)[0]
    top_semantic = sorted(semantic_indices, key=lambda i: similarities[i], reverse=True)[:5]
    matched_semantic = {node_labels[i] for i in top_semantic}

    matched = list(matched_substring | matched_semantic)[:5]

    if not matched:
        return "No matching entities found in knowledge graph."

    lines: list[str] = []
    for node in matched:
        if node not in G:
            continue
        successors = list(G.successors(node))
        predecessors = list(G.predecessors(node))
        if successors:
            lines.append(f"{node} -> {', '.join(str(n) for n in successors[:4])}")
        if predecessors:
            lines.append(f"{', '.join(str(n) for n in predecessors[:4])} -> {node}")

    return "\n".join(lines) if lines else "Entities found but no relationships retrieved."


def graph_retrieve_node(state) -> dict:
    context = _graph_search(state.query)
    accumulated = _append_context(state.accumulated_context, context)
    return {"graph_context": context, "accumulated_context": accumulated}