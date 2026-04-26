"""
classify.py — Query classification node. Carried over from v5.5 unchanged.
"""

from __future__ import annotations

from compressor_longtermrag.nodes.clients import classifier_llm


def classify_node(state) -> dict:
    prompt = f"""You are a query router for an air compressor maintenance assistant.

Classify the following query into exactly one of these types:
- spec        : asks for a specific parameter value, threshold, setting, or service interval
- procedure   : asks for steps, instructions, or narrative guidance
- relation    : asks about relationships between components
- hybrid      : requires BOTH a structured value AND either component relationships
                OR procedural narrative

Reply with a single word — the query type.

Query: {state.query}
"""
    result = classifier_llm.invoke(prompt)
    query_type = result.content.strip().lower()
    if query_type not in ("spec", "procedure", "relation", "hybrid"):
        query_type = "procedure"
    return {"query_type": query_type}
