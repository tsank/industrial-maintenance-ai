"""
synthesise.py — Context assembly node. Carried over from v5.5 with one addition:
if retrieved_memories is non-empty, past Q&A pairs are prepended as context.
"""

from __future__ import annotations

from compressor_longtermrag.nodes.clients import EMPTY_SIGNALS


def _is_useful(value: str) -> bool:
    if not value:
        return False
    return not any(value.startswith(signal) for signal in EMPTY_SIGNALS)


def synthesise_node(state) -> dict:
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

    return {"context": context}