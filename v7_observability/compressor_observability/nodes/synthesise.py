"""
synthesise.py — Context assembly node. Updated from v6 with the following changes:
1) Import from `compressor_observability` module and import traced_node for decorator
2) Create the decortor and add `span` to the node argument
3) Set span attributes
"""

from __future__ import annotations

from compressor_observability.nodes.clients import EMPTY_SIGNALS
from compressor_observability.observability import traced_node


def _is_useful(value: str) -> bool:
    if not value:
        return False
    return not any(value.startswith(signal) for signal in EMPTY_SIGNALS)


@traced_node("node.synthesise")
def synthesise_node(state, span) -> dict:
    # --- Memory context block (v6 addition) ---
    memory_block = ""
    if state.retrieved_memories:
        lines = ["[Relevant past interactions]"]
        for entry in state.retrieved_memories:
            lines.append(f"Q: {entry.query}")
            lines.append(f"A: {entry.answer}")
            lines.append("")
        memory_block = "\n".join(lines)

    # --- Assemble retrieval context (unchanged from v5.5) ---
    if state.accumulated_context and _is_useful(state.accumulated_context):
        retrieval_context = state.accumulated_context
    else:
        parts: list[str] = []
        if _is_useful(state.sql_context):
            parts.append(f"[Structured data]\n{state.sql_context}")
        if _is_useful(state.vector_context):
            parts.append(f"[Narrative/procedural text]\n{state.vector_context}")
        if _is_useful(state.graph_context):
            parts.append(f"[Component relationships]\n{state.graph_context}")
        retrieval_context = "\n\n---\n\n".join(parts) if parts else "No context retrieved."

    # --- Combine memory block with retrieval context ---
    if memory_block:
        context = f"{memory_block}\n\n---\n\n{retrieval_context}"
    else:
        context = retrieval_context

    span.set_attribute("memory_injected", bool(state.retrieved_memories))
    span.set_attribute("context_length", len(context))

    return {"context": context}