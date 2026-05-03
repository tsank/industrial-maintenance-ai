"""
retrieve_graph.py — NetworkX graph retrieval node. Following are the changes with respect to v6:
1) Imports updated to come from `compressor_observability` module
2) Set attributes to span inside `graph_retrieve_node():
    - Add `@traced_node("node.retrieve_graph")
    - Add `span` as an attribute
    - Create `span.set_attribute()
3) Add child_span in helper function
    - Import `get_tracer` `from compressor_observability` module
    - Put the logic inside running `with` block
    - Add `child_span.set_attribute() to track counts of substring, semantic and overall matches
"""

from __future__ import annotations

import numpy as np

from compressor_observability.nodes.clients import (
    embeddings, G, node_labels, node_matrix, SEMANTIC_THRESHOLD,
)
from compressor_observability.observability import traced_node


def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"


def _graph_search(query: str) -> str:
    from compressor_observability.observability import get_tracer
    tracer = get_tracer()

    with tracer.start_as_current_span("helper.graph_search") as child_span:
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
        child_span.set_attribute("entities_matched", len(matched))
        child_span.set_attribute("substring_matches", len(matched_substring))
        child_span.set_attribute("semantic_matches", len(matched_semantic))

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


@traced_node("node.retrieve_graph")
def graph_retrieve_node(state, span) -> dict:
    context = _graph_search(state.query)
    accumulated = _append_context(state.accumulated_context, context)
    span.set_attribute("context_length", len(context))
    return {"graph_context": context, "accumulated_context": accumulated}