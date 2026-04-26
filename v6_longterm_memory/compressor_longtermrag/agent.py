"""
agent.py — LangGraph StateGraph for long-term memory RAG.

New in v6 vs v5.5:
  - PostgresSaver checkpointer for cross-session state persistence
  - session_id generated via uuid4 if not supplied by caller
  - memory_retrieval_node runs before classify — fetches top-k past episodes
  - memory_storage_node runs after judge accepts — persists episode to PostgreSQL
  - synthesise_node prepends retrieved memories to context (see synthesise.py)

Graph structure:
  memory_retrieval → classify → [db/vector/graph/hybrid]_retrieve
  → synthesise → generate → judge → (retry via re_retrieve | end via memory_storage)

Exit conditions (unchanged from v5.5):
  - Judge says 'good'                                          → memory_storage → end
  - Judge says 'insufficient' AND attempts exhausted          → memory_storage → end
  - Judge says 'insufficient' AND score < threshold           → memory_storage → end
  - Judge says 'insufficient' AND score declining             → memory_storage → end
  - Judge says 'insufficient' AND parse error                 → retry
  - Judge says 'insufficient' AND score stable/improving      → retry
  - Judge says 'insufficient' AND no trend yet                → retry
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from compressor_longtermrag.state import AgentState
from compressor_longtermrag.nodes.memory_retrieval import memory_retrieval_node
from compressor_longtermrag.nodes.memory_storage import memory_storage_node
from compressor_longtermrag.nodes.classify import classify_node
from compressor_longtermrag.nodes.retrieve_spec import db_retrieve_node
from compressor_longtermrag.nodes.retrieve_vector import vector_retrieve_node
from compressor_longtermrag.nodes.retrieve_graph import graph_retrieve_node
from compressor_longtermrag.nodes.retrieve_hybrid import hybrid_retrieve_node
from compressor_longtermrag.nodes.synthesise import synthesise_node
from compressor_longtermrag.nodes.judge import judge_node

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Generate node (inline — identical to v5.5)
# ---------------------------------------------------------------------------

from compressor_longtermrag.nodes.clients import llm


def generate_node(state) -> dict:
    prompt = f"""You are a maintenance assistant for an Atlas Copco GA5 air compressor.
Answer the question using only the provided context. Be concise and precise.
If the context does not contain enough information, say so.

Context:
{state.context}

Question: {state.query}
"""
    result = llm.invoke(prompt)
    previous_answer = state.answer
    answer = result.content.strip()
    return {"answer": answer, "previous_answer": previous_answer}


# ---------------------------------------------------------------------------
# Re-retrieve node (inline — identical to v5.5)
# ---------------------------------------------------------------------------

from compressor_longtermrag.nodes.clients import (
    classifier_llm, embeddings, collection, pg_conn,
    QUERY_TYPE_TO_STORE,
)
from compressor_longtermrag.nodes.retrieve_spec import _generate_sql, _execute_sql
from compressor_longtermrag.nodes.retrieve_graph import _graph_search


def _select_fallback_store(query_type: str, evaluation: str) -> str:
    original_store = QUERY_TYPE_TO_STORE.get(query_type, "chromadb")
    prompt = f"""An AI assistant tried to answer a maintenance question but the answer was incomplete.

Original query type: {query_type}
Original store used: {original_store}
Judge evaluation — what is missing: {evaluation}

Which store is most likely to contain the missing information?

Stores available:
- postgresql  : exact parameter values, thresholds, service intervals, pressure settings
- chromadb    : narrative procedures, safety instructions, step-by-step text
- networkx    : component relationships, system dependencies, entity connections

Reply with exactly one word: postgresql, chromadb, or networkx.
"""
    result = classifier_llm.invoke(prompt)
    store = result.content.strip().lower()
    if store not in ("postgresql", "chromadb", "networkx"):
        store = original_store
    return store


def re_retrieve_node(state) -> dict:
    targeted_query = (
        f"{state.query} — specifically: {state.evaluation}"
        if state.evaluation else state.query
    )
    fallback_store = _select_fallback_store(state.query_type, state.evaluation)

    if fallback_store == "postgresql":
        sql = _generate_sql(targeted_query)
        new_content = _execute_sql(sql)
    elif fallback_store == "chromadb":
        query_vec = embeddings.embed_query(targeted_query)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=4,
            include=["documents"],
        )
        docs = results["documents"][0] if results["documents"] else []
        new_content = "\n\n".join(docs) if docs else "No relevant passages found."
    else:
        new_content = _graph_search(targeted_query)

    accumulated = (
        f"{state.accumulated_context}\n\n---\n\n{new_content}"
        if state.accumulated_context else new_content
    )
    return {"accumulated_context": accumulated, "fallback_store": fallback_store}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> str:
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

def build_graph(checkpointer) -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("memory_retrieval", memory_retrieval_node)
    g.add_node("classify", classify_node)
    g.add_node("db_retrieve", db_retrieve_node)
    g.add_node("vector_retrieve", vector_retrieve_node)
    g.add_node("graph_retrieve", graph_retrieve_node)
    g.add_node("hybrid_retrieve", hybrid_retrieve_node)
    g.add_node("synthesise", synthesise_node)
    g.add_node("generate", generate_node)
    g.add_node("judge", judge_node)
    g.add_node("re_retrieve", re_retrieve_node)
    g.add_node("memory_storage", memory_storage_node)

    g.set_entry_point("memory_retrieval")
    g.add_edge("memory_retrieval", "classify")

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
    g.add_edge("generate", "judge")

    g.add_conditional_edges(
        "judge",
        should_retry,
        {
            "retry": "re_retrieve",
            "end": "memory_storage",
        },
    )

    g.add_edge("re_retrieve", "synthesise")
    g.add_edge("memory_storage", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Checkpointer and app
# ---------------------------------------------------------------------------

import psycopg

_pg_conn_str = (
    f"host={os.environ['PG_HOST']} "
    f"port={os.environ.get('PG_PORT', '5432')} "
    f"dbname={os.environ['PG_DB']} "
    f"user={os.environ['PG_USER']} "
    f"password={os.environ['PG_PASSWORD']}"
)

_psycopg_conn = psycopg.connect(_pg_conn_str, autocommit=True)
checkpointer = PostgresSaver(_psycopg_conn)
checkpointer.setup()

app = build_graph(checkpointer)


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------

def run(
    query: str,
    session_id: str | None = None,
    max_attempts: int = 3,
    min_score_threshold: float = 0.3,
) -> AgentState:
    """
    Run the agent for a given query.
    If session_id is None, a new uuid4 session is created.
    Pass the same session_id across calls to maintain cross-session memory.
    """
    sid = session_id or f"session_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": sid}}
    result = app.invoke(
        {
            "query": query,
            "session_id": sid,
            "max_attempts": max_attempts,
            "min_score_threshold": min_score_threshold,
        },
        config=config,
    )
    return AgentState(**result)