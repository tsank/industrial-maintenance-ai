"""
retrieve_hybrid.py — Hybrid retrieval node (SQL + graph). Following changes are done as compared to v6:
1) Import from `compressor_observability` module and add import of traced_node for the decorator
2) Add decorator @traced_node("node.retrieve_hybrid") to hybrid_retrieve_node()
    - Pass on span created in the decorator function hybrid_retrieve_node
    - Create span.set_attribute() to trace sql context and graph context
"""

from __future__ import annotations

from compressor_observability.nodes.clients import classifier_llm
from compressor_observability.nodes.retrieve_spec import _generate_sql, _execute_sql
from compressor_observability.nodes.retrieve_graph import _graph_search
from compressor_observability.observability import traced_node

def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"


def _extract_sql_question(query: str) -> str:
    prompt = f"""Extract only the part of this question that can be answered from structured tabular data
(parameters, thresholds, service intervals, pressure settings).
Remove any parts about component relationships, procedures, or steps.
Return only the extracted question, nothing else.

Question: {query}
"""
    result = classifier_llm.invoke(prompt)
    return result.content.strip()

@traced_node("node.retrieve_hybrid")
def hybrid_retrieve_node(state, span) -> dict:
    sql_query = _extract_sql_question(state.query)
    sql = _generate_sql(sql_query)
    sql_rows = _execute_sql(sql)
    graph_ctx = _graph_search(state.query)
    new_content = f"{sql_rows}\n\n{graph_ctx}"
    accumulated = _append_context(state.accumulated_context, new_content)
    span.set_attribute("sql_context_length", len(sql_rows))
    span.set_attribute("graph_context_length", len(graph_ctx))
    return {
        "sql_context": sql_rows,
        "graph_context": graph_ctx,
        "accumulated_context": accumulated,
    }
