"""
classify.py — Query classification node. Following changes made from v6:
1) Importing from compressor_observability.nodes.clients
2) Adding decorator @traced_node("node.classify")
3) Passing two attributes to definition of `classify_node(state, span)
4) Setting attribute for span `span.set_attribute(...)
"""

from __future__ import annotations

from compressor_observability.nodes.clients import classifier_llm
from compressor_observability.observability import traced_node

@traced_node("node.classify")
def classify_node(state, span) -> dict:
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
    span.set_attribute("query_type", query_type)
    return {"query_type": query_type}
