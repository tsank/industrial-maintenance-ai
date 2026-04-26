"""
test_agent.py — Standard invoke tests extended for v6 memory fields.

Verifies:
- Agent returns a non-empty answer
- session_history is populated after a run
- memory_injected flag is set
- verdict is valid
"""

from __future__ import annotations

import uuid
import pytest

from compressor_longtermrag.agent import run
from compressor_longtermrag.state import AgentState


def _make_session_id() -> str:
    return f"test_agent_{uuid.uuid4().hex}"


def test_spec_query():
    state = run(
        query="What is the maximum working pressure of the GA5?",
        session_id=_make_session_id(),
    )
    assert isinstance(state, AgentState)
    assert state.answer
    assert state.verdict in ("good", "insufficient")
    assert state.memory_injected is True
    assert isinstance(state.session_history, list)
    assert len(state.session_history) == 1


def test_procedure_query():
    state = run(
        query="What are the steps to check the oil level?",
        session_id=_make_session_id(),
    )
    assert state.answer
    assert state.memory_injected is True
    # session_history populated only if score >= min_score_threshold
    assert len(state.session_history) == (1 if state.score >= state.min_score_threshold else 0)


def test_relation_query():
    state = run(
        query="What components are connected to the oil separator?",
        session_id=_make_session_id(),
    )
    assert state.answer
    assert len(state.session_history) == (1 if state.score >= state.min_score_threshold else 0)


def test_session_history_entry_fields():
    state = run(
        query="What is the oil capacity of the GA5?",
        session_id=_make_session_id(),
    )
    assert len(state.session_history) == 1
    entry = state.session_history[0]
    assert entry.query
    assert entry.answer
    assert entry.session_id
    assert entry.store_used in ("spec", "procedure", "relation", "hybrid")