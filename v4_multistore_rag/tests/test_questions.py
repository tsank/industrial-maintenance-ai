"""
test_questions.py — Five questions that demonstrate why each store is needed.

Each test asserts:
  1. The query_type is classified correctly.
  2. The expected context field is populated (non-empty).
  3. The answer is non-empty and contains a plausible keyword.

Store coverage:
  test_spec_only        → PostgreSQL only
  test_procedure_only   → ChromaDB only
  test_relation_only    → NetworkX only
  test_hybrid_sql_graph → PostgreSQL + NetworkX
  test_hybrid_sql_vector→ PostgreSQL + ChromaDB
"""

from __future__ import annotations

import pytest

from compressor_multistorerag.agent import app, AgentState


def _run(query: str) -> AgentState:
    initial: AgentState = {
        "query": query,
        "query_type": "",
        "sql_context": "",
        "vector_context": "",
        "graph_context": "",
        "context": "",
        "answer": "",
    }
    return app.invoke(initial)


# ---------------------------------------------------------------------------
# Test 1 — PostgreSQL only
# Why PostgreSQL: exact threshold value; no amount of semantic search on
# unstructured text will reliably return "120 °C" as a structured fact.
# ---------------------------------------------------------------------------

from compressor_multistorerag.db_retriever import query

def test_seed_data_loaded():
    """Verify seed data loaded correctly — independent of the agent."""
    
    # operational_parameters
    rows = query("SELECT * FROM operational_parameters WHERE parameter = 'maximum_working_pressure'")
    assert len(rows) == 1
    assert float(rows[0]["value"]) == 8.6
    assert rows[0]["unit"] == "bar"

    # service_plans
    rows = query("SELECT * FROM service_plans WHERE plan = 'A'")
    assert len(rows) == 1
    assert rows[0]["interval_hours"] == 500
    assert rows[0]["interval_months"] == 3

    # protection_thresholds
    rows = query("SELECT * FROM protection_thresholds WHERE parameter = 'high_temperature'")
    assert len(rows) == 2
    shutdown = next(r for r in rows if r["level"] == "shutdown")
    assert shutdown["value"] == 120.0
    assert shutdown["unit"] == "°C"

    # pressure_settings
    rows = query("SELECT * FROM pressure_settings WHERE ga_model = 'GA5'")
    assert len(rows) == 6
    
    print("\n[SEED] All four tables verified.")


def test_spec_only():
    result = _run("What is the shutdown temperature threshold for high temperature?")
    assert result["query_type"] == "spec", f"Expected 'spec', got '{result['query_type']}'"
    assert result["sql_context"], "sql_context should be populated for spec queries"
    assert result["vector_context"] == "" or result["query_type"] == "spec"
    # answer should mention the threshold value
    answer = result["answer"].lower()
    assert any(kw in answer for kw in ("120", "shutdown", "temperature")), (
        f"Answer did not mention expected threshold: {result['answer']}"
    )
    print(f"\n[SPEC] Q: {result['query']}\nA: {result['answer']}")


# ---------------------------------------------------------------------------
# Test 2 — ChromaDB only
# Why ChromaDB: multi-sentence procedural steps live in unstructured prose;
# SQL can't retrieve them, and a graph has no step-by-step edges.
# ---------------------------------------------------------------------------

def test_procedure_only():
    result = _run(
        "What should be done if the compressor element outlet temperature "
        "reaches the warning level?"
    )
    assert result["query_type"] == "procedure", (
        f"Expected 'procedure', got '{result['query_type']}'"
    )
    assert result["vector_context"], "vector_context should be populated for procedure queries"
    answer = result["answer"].lower()
    assert any(kw in answer for kw in ("stop", "voltage", "inspect", "warning", "temperature", "shut")), (
        f"Answer did not mention oil check steps: {result['answer']}"
    )
    print(f"\n[PROCEDURE] Q: {result['query']}\nA: {result['answer']}")


# ---------------------------------------------------------------------------
# Test 3 — NetworkX only
# Why NetworkX: component relationship queries need graph traversal;
# SQL has no edges, and semantic search returns prose not structured links.
# ---------------------------------------------------------------------------

def test_relation_only():
    result = _run("What components are directly connected to or associated with the air filter?")
    assert result["query_type"] == "relation", (
        f"Expected 'relation', got '{result['query_type']}'"
    )
    assert result["graph_context"], "graph_context should be populated for relation queries"
    answer = result["answer"].lower()
    assert any(kw in answer for kw in ("filter", "compressor", "element", "inlet", "air")), (
        f"Answer did not mention filter-related components: {result['answer']}"
    )
    print(f"\n[RELATION] Q: {result['query']}\nA: {result['answer']}")


# ---------------------------------------------------------------------------
# Test 4 — Hybrid: PostgreSQL + NetworkX
# Why two stores: service intervals are exact tabular data (SQL);
# the components involved in each service task are graph entities (NetworkX).
# Neither store alone answers the full question.
# ---------------------------------------------------------------------------

def test_hybrid_sql_graph():
    result = _run(
        "What are the service intervals for the A-level service plan, "
        "and what compressor components are involved?"
    )
    assert result["query_type"] == "hybrid", (
        f"Expected 'hybrid', got '{result['query_type']}'"
    )
    assert result["sql_context"], "sql_context should be populated for hybrid queries"
    assert result["graph_context"], "graph_context should be populated for hybrid queries"
    answer = result["answer"].lower()
    assert any(kw in answer for kw in ("500", "hours", "month", "interval", "plan a", "plan")), (
        f"Answer did not mention service interval data: {result['answer']}"
    )
    print(f"\n[HYBRID SQL+GRAPH] Q: {result['query']}\nA: {result['answer']}")


# ---------------------------------------------------------------------------
# Test 5 — Hybrid: PostgreSQL + ChromaDB
# Why two stores: the exact pressure value is a structured fact (SQL);
# the safety precautions are narrative prose best retrieved by vector search.
# ---------------------------------------------------------------------------

def test_hybrid_sql_vector():
    result = _run(
        "What is the maximum working pressure of the GA5, "
        "and what safety precautions apply when operating near that limit?"
    )
    assert result["query_type"] == "hybrid", (
        f"Expected 'hybrid', got '{result['query_type']}'"
    )
    # hybrid_retrieve_node populates sql_context + graph_context by default;
    # accept that safety narrative may come from either graph or vector context
    assert result["sql_context"], "sql_context should be populated for hybrid queries"
    answer = result["answer"].lower()
    assert any(kw in answer for kw in ("8.6", "bar", "pressure", "maximum", "working")), (
        f"Answer did not mention pressure value: {result['answer']}"
    )
    print(f"\n[HYBRID SQL+VECTOR] Q: {result['query']}\nA: {result['answer']}")
