"""
agent.py — LangGraph StateGraph for self-evaluating agentic RAG.

New in v5 vs v4:
  - AgentState migrated from TypedDict to Pydantic for runtime type enforcement
  - judge_node evaluates answer quality after generation
  - re_retrieve_node uses judge feedback for targeted re-retrieval
  - Graph contains a cycle — retrieve → synthesise → generate → judge → retry
  - should_retry() implements score trend early stopping

Exit conditions:
  - Judge says 'good'                                                    → exit with current answer
  - Judge says 'insufficient' AND attempts exhausted                     → exit with current answer
  - Judge says 'insufficient' AND valid JSON AND score < threshold       → exit with current answer
  - Judge says 'insufficient' AND parse error                            → retry unconditionally
  - Judge says 'insufficient' AND valid JSON AND score declining         → exit with previous answer
  - Judge says 'insufficient' AND valid JSON AND score stable/improving  → retry
  - Judge says 'insufficient' AND valid JSON AND no trend yet            → retry
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from compressor_selfevalrag.pipeline import (
    classify_node,
    db_retrieve_node,
    vector_retrieve_node,
    graph_retrieve_node,
    hybrid_retrieve_node,
    synthesise_node,
    generate_node,
    judge_node,
    re_retrieve_node,
)


# ---------------------------------------------------------------------------
# State schema — Pydantic for runtime type enforcement
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    # Core query
    query: str
    query_type: str = ""

    # Attempt management
    attempt: int = 0
    max_attempts: int = 3

    # Retrieval context fields — accumulated across retries
    sql_context: str = ""
    vector_context: str = ""
    graph_context: str = ""
    accumulated_context: str = ""   # grows with each retry
    context: str = ""               # current attempt's assembled context

    # Answer fields
    answer: str = ""
    previous_answer: str = ""       # answer from attempt before current

    # Judge fields
    verdict: str = ""                  # "good" | "insufficient"
    evaluation: str = ""               # judge's description of what is missing
    score: float = 0.0                 # judge's score for current answer
    score_trend: list[float] = Field(default_factory=list)
    min_score_threshold: float = 0.3   # exit if score below this - content likely not in knowledge base
    judge_parse_error: bool = False    # True if last judge response was malformed JSON


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> str:
    """Route to the correct retrieval node based on query_type."""
    return {
        "spec": "db_retrieve",
        "procedure": "vector_retrieve",
        "relation": "graph_retrieve",
        "hybrid": "hybrid_retrieve",
    }.get(state.query_type, "vector_retrieve")


def should_retry(state: AgentState) -> str:
    if state.verdict == "good":
        return "end"

    if state.attempt >= state.max_attempts:
        return "end"

    # Both threshold and trend checks only apply to valid judge responses
    if not state.judge_parse_error:
        if state.score < state.min_score_threshold:
            return "end"

        if len(state.score_trend) >= 2:
            if state.score_trend[-1] < state.score_trend[-2]:
                return "end"

    return "retry"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # --- Nodes ---
    g.add_node("classify", classify_node)
    g.add_node("db_retrieve", db_retrieve_node)
    g.add_node("vector_retrieve", vector_retrieve_node)
    g.add_node("graph_retrieve", graph_retrieve_node)
    g.add_node("hybrid_retrieve", hybrid_retrieve_node)
    g.add_node("synthesise", synthesise_node)
    g.add_node("generate", generate_node)
    g.add_node("judge", judge_node)
    g.add_node("re_retrieve", re_retrieve_node)

    # --- Entry point ---
    g.set_entry_point("classify")

    # --- Classification fan-out ---
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

    # --- First retrieval convergence ---
    for retriever in ("db_retrieve", "vector_retrieve", "graph_retrieve", "hybrid_retrieve"):
        g.add_edge(retriever, "synthesise")

    # --- Linear path to judge ---
    g.add_edge("synthesise", "generate")
    g.add_edge("generate", "judge")

    # --- Conditional edge from judge — the cycle ---
    g.add_conditional_edges(
        "judge",
        should_retry,
        {
            "retry": "re_retrieve",
            "end": END,
        },
    )

    # --- Retry cycle ---
    g.add_edge("re_retrieve", "synthesise")

    return g.compile()


app = build_graph()
