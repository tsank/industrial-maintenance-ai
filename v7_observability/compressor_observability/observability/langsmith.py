"""
observability/langsmith.py

Configures LangSmith tracing by setting the environment variables
that LangChain reads automatically at import time.

Called once from nodes/clients.py at module load.
If LANGCHAIN_API_KEY is absent, tracing is silently skipped —
the graph runs identically without it.

LangSmith captures automatically (no instrumentation code needed):
  - Every LLM call: prompt, response, token usage, latency
  - LangGraph node transitions
  - Chain inputs and outputs
"""

import os
import logging

logger = logging.getLogger(__name__)


def setup_langsmith() -> None:
    """
    Activate LangSmith tracing if credentials are present in the environment.

    Required .env keys:
        LANGCHAIN_API_KEY      — your LangSmith API key
        LANGCHAIN_PROJECT      — project name shown in LangSmith UI
        LANGCHAIN_TRACING_V2   — must be "true"
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "industrial-maintenance-ai")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()

    if not api_key:
        logger.info("LANGCHAIN_API_KEY not set — LangSmith tracing disabled")
        # Explicitly disable so LangChain does not attempt to connect
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return

    # Set variables LangChain reads automatically
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGCHAIN_API_KEY"] = api_key

    logger.info("LangSmith tracing enabled → project: %s", project)


def is_langsmith_enabled() -> bool:
    """
    Returns True if LangSmith tracing is active.
    Used by the Gradio observability tab to decide whether to show the link.
    """
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def get_langsmith_project_url() -> str | None:
    """
    Returns the LangSmith project URL for display in the Gradio UI.
    Returns None if LangSmith is not configured.
    """
    project = os.getenv("LANGCHAIN_PROJECT")
    if not project or not is_langsmith_enabled():
        return None
    return f"https://smith.langchain.com/projects/{project}"
