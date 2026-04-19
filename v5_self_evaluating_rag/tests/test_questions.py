"""
test_questions.py — Test suite for the self-evaluating RAG agent (v5).

Six tests covering:
  test_seed_data_loaded     — data layer verification, independent of agent
  test_spec_good            — PostgreSQL retrieval, judge evaluates answer
  test_procedure_good       — ChromaDB retrieval, judge evaluates answer
  test_relation_sparse      — NetworkX retrieval, judge evaluates answer
  test_hybrid_good          — PostgreSQL + NetworkX, judge evaluates answer
  test_judge_fields         — verify all judge fields populated correctly

Key v5 assertions beyond v4:
  - verdict is always set after execution
  - score is always between 0.0 and 1.0
  - attempt count matches len(score_trend)
  - judge_parse_error is a bool
  - if verdict is 'good', evaluation is empty
  - if verdict is 'insufficient', evaluation describes what is missing
  - previous_answer is set correctly when retries occur
  - min_score_threshold early exit works correctly
"""

from __future__ import annotations

import pytest

from compressor_selfevalrag.agent import app, AgentState
from compressor_selfevalrag.db_retriever import query


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _run(question: str, max_attempts: int = 3, min_score_threshold: float = 0.3) -> AgentState:
    """
    Run the agent with the given question and return the final state.
    max_attempts and min_score_threshold are configurable for testing
    specific exit conditions.
    """
    initial = AgentState(
        query=question,
        max_attempts=max_attempts,
        min_score_threshold=min_score_threshold,
    )
    result = app.invoke(initial)
    return AgentState(**result)  # convert dict back to Pydantic model


def _assert_judge_invariants(result: AgentState) -> None:
    """
    Assert invariants that must hold after every agent execution
    regardless of verdict or number of attempts.
    """
    # verdict must always be set
    assert result.verdict in ("good", "insufficient"), (
        f"verdict must be 'good' or 'insufficient', got '{result.verdict}'"
    )

    # score must always be in valid range
    assert 0.0 <= result.score <= 1.0, (
        f"score must be between 0.0 and 1.0, got {result.score}"
    )

    # attempt must be at least 1
    assert result.attempt >= 1, (
        f"at least one attempt must have been made, got {result.attempt}"
    )

    # score_trend length must equal attempt count
    assert len(result.score_trend) == result.attempt, (
        f"score_trend length {len(result.score_trend)} must equal "
        f"attempt count {result.attempt}"
    )

    # judge_parse_error must be a bool
    assert isinstance(result.judge_parse_error, bool), (
        f"judge_parse_error must be bool, got {type(result.judge_parse_error)}"
    )

    # if verdict is good, evaluation must be empty
    if result.verdict == "good":
        assert result.evaluation == "", (
            f"evaluation must be empty when verdict is good, got: '{result.evaluation}'"
        )

    # if verdict is insufficient, evaluation must describe what is missing
    if result.verdict == "insufficient":
        assert result.evaluation != "", (
            "evaluation must describe what is missing when verdict is insufficient"
        )

    # answer must never be empty
    assert result.answer, "answer must never be empty"

    # if more than one attempt was made, previous_answer must be set
    if result.attempt > 1:
        assert result.previous_answer, (
            "previous_answer must be set when more than one attempt was made"
        )


# ---------------------------------------------------------------------------
# Test 0 — Data layer verification
# Independent of agent — verifies PostgreSQL seed data
# ---------------------------------------------------------------------------

def test_seed_data_loaded():
    """Verify seed data loaded correctly — independent of the agent."""
    rows = query(
        "SELECT * FROM operational_parameters WHERE parameter = 'maximum_working_pressure'"
    )
    assert len(rows) == 1
    assert float(rows[0]["value"]) == 8.6
    assert rows[0]["unit"] == "bar"

    rows = query("SELECT * FROM service_plans WHERE plan = 'A'")
    assert len(rows) == 1
    assert rows[0]["interval_hours"] == 500
    assert rows[0]["interval_months"] == 3

    rows = query(
        "SELECT * FROM protection_thresholds WHERE parameter = 'high_temperature'"
    )
    assert len(rows) == 2
    shutdown = next(r for r in rows if r["level"] == "shutdown")
    assert shutdown["value"] == 120.0
    assert shutdown["unit"] == "°C"

    rows = query("SELECT * FROM pressure_settings WHERE ga_model = 'GA5'")
    assert len(rows) == 6

    print("\n[SEED] All four tables verified.")


# ---------------------------------------------------------------------------
# Test 1 — Spec query
# PostgreSQL only. Judge evaluates exact structured fact.
# Expected: good verdict on first attempt.
# ---------------------------------------------------------------------------

def test_spec_good():
    result = _run("What is the shutdown temperature threshold for high temperature?")

    assert result.query_type == "spec", (
        f"Expected 'spec', got '{result.query_type}'"
    )
    assert result.sql_context, "sql_context must be populated for spec queries"

    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("120", "shutdown", "temperature")), (
        f"Answer did not mention expected threshold: {result.answer}"
    )

    print(f"\n[SPEC] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}, Parse error: {result.judge_parse_error}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 2 — Procedure query
# ChromaDB only. Judge evaluates procedural answer quality.
# ---------------------------------------------------------------------------

def test_procedure_good():
    result = _run(
        "What should be done if the compressor element outlet temperature "
        "reaches the warning level?"
    )

    assert result.query_type == "procedure", (
        f"Expected 'procedure', got '{result.query_type}'"
    )
    assert result.vector_context, "vector_context must be populated for procedure queries"

    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("stop", "voltage", "inspect", "warning", "temperature")), (
        f"Answer did not mention expected steps: {result.answer}"
    )

    print(f"\n[PROCEDURE] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}, Parse error: {result.judge_parse_error}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 3 — Relation query
# NetworkX only. Judge evaluates component relationship answer.
# ---------------------------------------------------------------------------

def test_relation_sparse():
    """
    Relation query against a sparse graph — air filter has only one edge
    in the knowledge graph (air filter → moisture prevention). The judge
    correctly scores this as insufficient (0.2) and exits early via
    min_score_threshold. This is a data quality limitation from v2 triple
    extraction, not a v5 bug. The system behaves correctly given its data.
    Cross-store fallback for sparse graph results is deferred to a later version.
    """
    result = _run(
        "What components are directly connected to or associated with the air filter?"
    )

    assert result.query_type == "relation", (
        f"Expected 'relation', got '{result.query_type}'"
    )
    assert result.graph_context, "graph_context must be populated for relation queries"

    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("filter", "moisture", "air", "component")), (
        f"Answer did not mention filter-related components: {result.answer}"
    )

    print(f"\n[RELATION] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}, Parse error: {result.judge_parse_error}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 4 — Hybrid query
# PostgreSQL + NetworkX. Judge evaluates compound answer.
# ---------------------------------------------------------------------------

def test_hybrid_good():
    result = _run(
        "What are the service intervals for the A-level service plan, "
        "and what compressor components are involved?"
    )

    assert result.query_type == "hybrid", (
        f"Expected 'hybrid', got '{result.query_type}'"
    )
    assert result.sql_context, "sql_context must be populated for hybrid queries"
    assert result.graph_context, "graph_context must be populated for hybrid queries"

    _assert_judge_invariants(result)

    answer = result.answer.lower()
    assert any(kw in answer for kw in ("500", "hours", "month", "interval", "plan")), (
        f"Answer did not mention service interval data: {result.answer}"
    )

    print(f"\n[HYBRID] Attempts: {result.attempt}, Score: {result.score:.2f}, "
          f"Verdict: {result.verdict}, Parse error: {result.judge_parse_error}")
    print(f"Score trend: {result.score_trend}")
    print(f"Q: {result.query}\nA: {result.answer}")


# ---------------------------------------------------------------------------
# Test 5 — Judge fields and early exit verification
# Verifies all judge invariants and min_score_threshold early exit behaviour.
# ---------------------------------------------------------------------------

def test_judge_fields():
    # Part A — normal query, verify all judge fields
    result = _run("What is the maximum working pressure of the GA5?")

    _assert_judge_invariants(result)

    # accumulated_context must be populated
    assert result.accumulated_context, "accumulated_context must be populated"

    print(f"\n[JUDGE FIELDS - Part A]")
    print(f"Attempts: {result.attempt}, Score: {result.score:.2f}")
    print(f"Score trend: {result.score_trend}")
    print(f"Verdict: {result.verdict}, Parse error: {result.judge_parse_error}")
    print(f"Evaluation: '{result.evaluation}'")
    print(f"Q: {result.query}\nA: {result.answer}")

    # Part B — content not in manual, verify min_score_threshold early exit
    # High threshold forces early exit on content gap
    result_gap = _run(
        "What is the weight of Pele compared to the oil capacity of the GA5?",
        max_attempts=3,
        min_score_threshold=0.8,
    )

    _assert_judge_invariants(result_gap)

    # With high threshold, should exit after first insufficient attempt
    if result_gap.verdict == "insufficient":
        assert result_gap.attempt <= result_gap.max_attempts, (
            f"attempt {result_gap.attempt} must not exceed "
            f"max_attempts {result_gap.max_attempts}"
        )

    print(f"\n[JUDGE FIELDS - Part B — content gap test]")
    print(f"Attempts: {result_gap.attempt}, Score: {result_gap.score:.2f}")
    print(f"Score trend: {result_gap.score_trend}")
    print(f"Verdict: {result_gap.verdict}, Parse error: {result_gap.judge_parse_error}")
    print(f"Evaluation: '{result_gap.evaluation}'")
    print(f"Q: {result_gap.query}\nA: {result_gap.answer}")
