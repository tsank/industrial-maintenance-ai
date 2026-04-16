"""
agent.py — LangGraph StateGraph for multi-store agentic RAG.

Four retrieval paths:
  spec      → PostgreSQL only        (Text-to-SQL)
  procedure → ChromaDB only          (vector search + metadata filter)
  relation  → NetworkX only          (graph traversal)
  hybrid    → PostgreSQL + NetworkX  (two stores, two context fields)
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from compressor_multistorerag.pipeline import (
    classify_node,
    db_retrieve_node,
    vector_retrieve_node,
    graph_retrieve_node,
    hybrid_retrieve_node,
    synthesise_node,
    generate_node,
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    query: str
    query_type: str       # spec | procedure | relation | hybrid
    sql_context: str      # populated by db_retrieve or hybrid_retrieve
    vector_context: str   # populated by vector_retrieve or hybrid_retrieve
    graph_context: str    # populated by graph_retrieve or hybrid_retrieve
    context: str          # merged by synthesise_node
    answer: str           # produced by generate_node


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> str:
    """Return the next node name based on query_type."""
    return {
        "spec": "db_retrieve",
        "procedure": "vector_retrieve",
        "relation": "graph_retrieve",
        "hybrid": "hybrid_retrieve",
    }.get(state["query_type"], "vector_retrieve")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("classify", classify_node)
    g.add_node("db_retrieve", db_retrieve_node)
    g.add_node("vector_retrieve", vector_retrieve_node)
    g.add_node("graph_retrieve", graph_retrieve_node)
    g.add_node("hybrid_retrieve", hybrid_retrieve_node)
    g.add_node("synthesise", synthesise_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_query,
        {
            "db_retrieve": "db_retrieve",
            "vector_retrieve": "vector_retrieve",
            "graph_retrieve": "graph_retrieve",
            "hybrid_retrieve": "hybrid_retrieve",
        },
    )

    for retriever in ("db_retrieve", "vector_retrieve", "graph_retrieve", "hybrid_retrieve"):
        g.add_edge(retriever, "synthesise")

    g.add_edge("synthesise", "generate")
    g.add_edge("generate", END)

    return g.compile()


app = build_graph()
