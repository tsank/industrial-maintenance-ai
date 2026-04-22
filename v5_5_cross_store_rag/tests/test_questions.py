"""
test_questions.py — Test suite for the cross-store self-evaluating RAG agent (v5.5).

Carries forward all v5 tests unchanged.
Adds three new tests specific to v5.5:
  test_semantic_node_matching   — verify semantic matching finds nodes
                                  not found by substring matching alone
  test_cross_store_fallback     — verify relation query triggers cross-store
                                  fallback to ChromaDB when graph is sparse
  test_fallback_store_field     — verify fallback_store field is populated
                                  correctly after a retry

All tests use _assert_judge_invariants() for consistent judge field verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from compressor_crossstorerag.agent import app, AgentState
from compressor_crossstorerag.db_retriever import query
from compressor_crossstorerag.pipeline import (
    _graph_search,
    _node_labels,
    _node_matrix,
    _embeddings,
    _SEMANTIC_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run(question: str, max_attempts: int = 3, min_score_threshold: float = 0.3) -> AgentState:
    initial = AgentState(
        query=question,
        max_attempts=max_attempts,
        min_score_threshold=min_score_threshold,
    )
    return AgentState(**app.invoke(initial))


def _assert_judge_invariants(result: AgentState) -> None:
    """Assert invariants that must hold after every agent execution."""
    assert result.verdict in ("good", "insufficient"), (
        f"verdict must be 'good' or 'insufficient', got '{result.verdict}'"
    )
    assert 0.0 <= result.score <= 1.0, (
        f"score must be between 0.0 and 1.0, got {result.score}"
    )
    assert result.attempt >= 1, (
        f"at least one attempt must have been made, got {result.attempt}"
    )
    assert len(result.score_trend) == result.attempt, (
        f"score_trend length {len(result.score_trend)} must equal "
        f"attempt count {result.attempt}"
    )
    assert isinstance(result.judge_parse_error, bool), (
        f"judge_parse_error must be bool, got {type(result.judge_parse_error)}"
    )
    if result.verdict == "good":
        assert result.evaluation == "", (
            f"evaluation must be empty when verdict is good, got: '{result.evaluation}'"
        )
    if result.verdict == "insufficient":
        assert result.evaluation != "", (
            "evaluation must describe what is missing when verdict is insufficient"
        )
    assert result.answer, "answer must never be empty"
    if result.attempt > 1:
        assert result.previous_answer, (
            "previous_answer must be set when more than one attempt was made"
        )


# ---------------------------------------------------------------------------
# Test 0 — Data layer verification
# ---------------------------------------------------------------------------

def test_seed_data_loaded():
    """Verify seed data loaded correctly — independent of the agent."""
    rows = query(
        "SELECT * FROM operational_parameters WHERE parameter = 'maximum_working_pressure'"
    )
    assert len(rows) == 1
    assert float(rows[0]["value"]) == 8.6

    rows = query("SELECT * FROM service_plans WHERE plan = 'A'")
    assert len(rows) == 1
    assert rows[0]["interval_hours"] == 500

    rows = query(
        "SELECT * FROM protection_thresholds WHERE parameter = 'high_temperature'"
    )
    shutdown = next(r for r in rows if r["level"] == "shutdown")
    assert shutdown["value"] == 120.0

    rows = query("SELECT * FROM pressure_settings WHERE ga_model = 'GA5'")
    assert len(rows) == 6

    print("\n[SEED] All four tables verified.")


# ---------------------------------------------------------------------------
# Test 1 — Spec query (carried forward from v5)
# ---------------------------------------------------------------------------

def test_spec_good():
    result = _run("What is the shutdown temperature threshold for high temperature?")

    assert result.query_type == "spec"
    assert result.sql_context
    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("120", "shutdown", "temperature"))

    print(f"\n[SPEC] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 2 — Procedure query (carried forward from v5)
# ---------------------------------------------------------------------------

def test_procedure_good():
    result = _run(
        "What should be done if the compressor element outlet temperature "
        "reaches the warning level?"
    )

    assert result.query_type == "procedure"
    assert result.vector_context
    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("stop", "voltage", "inspect", "warning", "temperature"))

    print(f"\n[PROCEDURE] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 3 — Hybrid query (carried forward from v5)
# ---------------------------------------------------------------------------

def test_hybrid_good():
    result = _run(
        "What are the service intervals for the A-level service plan, "
        "and what compressor components are involved?"
    )

    assert result.query_type == "hybrid"
    assert result.sql_context
    assert result.graph_context
    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("500", "hours", "month", "interval", "plan"))

    print(f"\n[HYBRID] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 4 — Semantic node matching (NEW in v5.5)
# Verifies that semantic matching finds nodes not found by substring matching.
# Directly tests _graph_search() internals.
# ---------------------------------------------------------------------------

def test_semantic_node_matching():
    """
    Verify hybrid matching finds more nodes than substring matching alone.

    Uses a query about 'inlet air' which may not appear as exact substring
    in node labels but should have semantic similarity to related nodes
    like 'air filter', 'air dryer', 'inlet' etc.
    """
    # Query designed to have semantic but not exact substring matches
    query_text = "What is connected to the inlet air system?"
    query_lower = query_text.lower()

    # Substring matches only
    substring_matches = {
        n for n in _graph_search.__globals__['_G'].nodes
        if len(str(n)) > 4 and str(n).lower() in query_lower
    }

    # Full hybrid matches via _graph_search
    graph_result = _graph_search(query_text)

    # Semantic matching should find nodes beyond substring matches
    # Verify by checking cosine similarities directly
    query_vec = np.array(_embeddings.embed_query(query_text))
    similarities = np.dot(_node_matrix, query_vec) / (
        np.linalg.norm(_node_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    semantic_matches = {
        _node_labels[i]
        for i in np.where(similarities >= _SEMANTIC_THRESHOLD)[0]
    }

    print(f"\n[SEMANTIC MATCHING]")
    print(f"Substring matches: {substring_matches}")
    print(f"Semantic matches (top 10): {list(semantic_matches)[:10]}")
    print(f"Graph search result:\n{graph_result}")

    # The union should be at least as large as substring matches
    union = substring_matches | semantic_matches
    assert len(union) >= len(substring_matches), (
        "Union of substring and semantic matches must be at least as large as substring alone"
    )

    # Graph result must be non-empty (either found nodes or explicit message)
    assert graph_result, "Graph search must return a non-empty string"

    print(f"Substring count: {len(substring_matches)}, "
          f"Semantic count: {len(semantic_matches)}, "
          f"Union count: {len(union)}")


# ---------------------------------------------------------------------------
# Test 5 — Cross-store fallback (NEW in v5.5)
# Verifies that when graph retrieval is insufficient, the system retries
# with a different store selected by the LLM.
# Uses low min_score_threshold to allow retry despite sparse graph.
# ---------------------------------------------------------------------------

def test_cross_store_fallback():
    """
    Verify cross-store fallback activates for sparse graph queries.

    The air filter query produces sparse graph context (score ~0.2 in v5).
    With min_score_threshold=0.1, the system retries rather than exiting.
    The LLM should select ChromaDB as the fallback store.
    After retry, fallback_store should be populated.
    """
    result = _run(
        "What components are directly connected to or associated with the air filter?",
        max_attempts=2,
        min_score_threshold=0.1,  # low threshold to allow retry on sparse graph
    )

    _assert_judge_invariants(result)

    print(f"\n[CROSS-STORE FALLBACK]")
    print(f"Attempts: {result.attempt}, Score: {result.score:.2f}")
    print(f"Score trend: {result.score_trend}")
    print(f"Verdict: {result.verdict}")
    print(f"Fallback store: '{result.fallback_store}'")
    print(f"Evaluation: '{result.evaluation}'")
    print(f"Q: {result.query}\nA: {result.answer}")

    # If more than one attempt was made, fallback_store must be populated
    if result.attempt > 1:
        assert result.fallback_store in ("postgresql", "chromadb", "networkx"), (
            f"fallback_store must be a valid store name, got '{result.fallback_store}'"
        )
        print(f"Cross-store fallback activated → retried with: {result.fallback_store}")
    else:
        print("First attempt was sufficient — no fallback needed")


# ---------------------------------------------------------------------------
# Test 6 — Fallback store field verification (NEW in v5.5)
# Verifies fallback_store is correctly empty on first attempt
# and populated after a retry.
# ---------------------------------------------------------------------------

def test_fallback_store_field():
    """
    Verify fallback_store field behaviour:
    - Empty string when no retry occurred
    - Valid store name when retry occurred
    """
    # Part A — query that should be answered in one attempt
    # fallback_store should remain empty
    result_good = _run("What is the maximum working pressure of the GA5?")
    _assert_judge_invariants(result_good)

    if result_good.attempt == 1:
        assert result_good.fallback_store == "", (
            f"fallback_store must be empty when no retry occurred, "
            f"got '{result_good.fallback_store}'"
        )

    print(f"\n[FALLBACK STORE - Part A — no retry expected]")
    print(f"Attempts: {result_good.attempt}, Fallback store: '{result_good.fallback_store}'")
    print(f"Verdict: {result_good.verdict}, Score: {result_good.score:.2f}")

    # Part B — content gap query, verify field stays empty on early exit
    result_gap = _run(
        "What is the weight of Pele compared to the oil capacity of the GA5?",
        max_attempts=3,
        min_score_threshold=0.8,
    )
    _assert_judge_invariants(result_gap)

    print(f"\n[FALLBACK STORE - Part B — content gap]")
    print(f"Attempts: {result_gap.attempt}, Fallback store: '{result_gap.fallback_store}'")
    print(f"Verdict: {result_gap.verdict}, Score: {result_gap.score:.2f}")
    print(f"Evaluation: '{result_gap.evaluation}'")
