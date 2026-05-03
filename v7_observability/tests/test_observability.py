"""
tests/test_observability.py

Verifies that the observability layer is correctly configured.

Test 1 — LangSmith config:
  Asserts that LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, and
  LANGCHAIN_TRACING_V2 are present and correctly set.
  Does not make a live API call.

Test 2 — Jaeger reachability:
  Asserts that the Jaeger UI HTTP endpoint is reachable.
  Requires docker-compose stack to be running.

Test 3 — OTel → Jaeger pipeline:
  Runs a real query, waits for the BatchSpanProcessor to flush,
  then queries the Jaeger API and asserts a trace exists for
  the service name.
"""

import os
import time
import requests
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

JAEGER_API = "http://localhost:16686/api"
SERVICE_NAME = "industrial-maintenance-ai-v7"


# ── Test 1: LangSmith configuration ─────────────────────────────────────────

def test_langsmith_env_configured():
    """LANGCHAIN_TRACING_V2 must be 'true' and project must be set."""
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    project = os.getenv("LANGCHAIN_PROJECT", "")

    assert tracing == "true", (
        "LANGCHAIN_TRACING_V2 is not 'true'. Check your .env file."
    )
    assert api_key, "LANGCHAIN_API_KEY is missing from .env"
    assert project, "LANGCHAIN_PROJECT is missing from .env"


# ── Test 2: Jaeger reachability ──────────────────────────────────────────────

def test_jaeger_ui_reachable():
    """Jaeger UI must respond at localhost:16686."""
    try:
        response = requests.get(f"{JAEGER_API}/services", timeout=5)
        assert response.status_code == 200, (
            f"Jaeger API returned {response.status_code}. "
            "Is the docker-compose stack running?"
        )
    except requests.exceptions.ConnectionError:
        pytest.fail(
            "Cannot reach Jaeger at localhost:16686. "
            "Run: docker-compose up -d"
        )


# ── Test 3: OTel → Jaeger trace pipeline ─────────────────────────────────────

def test_otel_trace_reaches_jaeger():
    """
    After a run(), a trace for the service must appear in Jaeger.
    The BatchSpanProcessor flushes on a schedule — we allow up to
    10 seconds for the trace to appear.
    """
    from compressor_observability import run

    # Trigger a run to generate spans
    run(query="What is the maximum operating pressure of the GA5?")

    # Wait for BatchSpanProcessor to flush
    deadline = time.time() + 10
    trace_found = False

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{JAEGER_API}/services",
                timeout=5,
            )
            services = resp.json().get("data", [])
            if SERVICE_NAME in services:
                trace_found = True
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    assert trace_found, (
        f"Service '{SERVICE_NAME}' did not appear in Jaeger within 10 seconds. "
        "Check that OTLP_ENDPOINT is set in .env and Jaeger is running."
    )
