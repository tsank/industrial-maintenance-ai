"""
test_memory_persistence.py — Verifies cross-session state persistence.

Invokes the agent twice with the same session_id and confirms that
the second run's session_history contains entries from the first run.
"""

from __future__ import annotations

import uuid
import pytest

from compressor_observability.agent import run


def test_cross_session_persistence():
    session_id = f"test_persist_{uuid.uuid4().hex}"

    # First invocation
    state1 = run(
        query="What is the maximum working pressure of the GA5?",
        session_id=session_id,
    )
    assert len(state1.session_history) == 1

    # Second invocation — same session_id
    state2 = run(
        query="What is the oil capacity of the GA5?",
        session_id=session_id,
    )
    # LangGraph checkpointer restores state — history should have both entries
    assert len(state2.session_history) == 2
    queries = [e.query for e in state2.session_history]
    assert "What is the maximum working pressure of the GA5?" in queries
    assert "What is the oil capacity of the GA5?" in queries