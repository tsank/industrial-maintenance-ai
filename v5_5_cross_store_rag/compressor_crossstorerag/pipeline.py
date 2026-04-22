"""
pipeline.py — Node implementations for the cross-store self-evaluating agent.

New in v5.5 vs v5:
  _graph_search()     — hybrid matching: substring union semantic
                        node embeddings pre-computed, loaded at module init
  _select_fallback_store() — LLM reads judge evaluation, selects best store
  re_retrieve_node()  — routes retry to LLM-selected store, not always same store

Everything else carried forward from v5 unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import networkx as nx
import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Client initialisation (once at import)
# ---------------------------------------------------------------------------

_CHROMA_HOST = os.environ["CHROMA_HOST"]
_CHROMA_PORT = int(os.environ["CHROMA_PORT"])
_CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "maintenance_manuals")

_chroma_client = chromadb.HttpClient(host=_CHROMA_HOST, port=_CHROMA_PORT)
_collection = _chroma_client.get_collection(_CHROMA_COLLECTION)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_pg_conn = psycopg2.connect(
    host=os.environ["PG_HOST"],
    port=int(os.environ.get("PG_PORT", 5432)),
    dbname=os.environ["PG_DB"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
)
_pg_conn.autocommit = True

# NetworkX graph
_GRAPH_PATH = Path(__file__).parent.parent / "data" / "graph.json"
with open(_GRAPH_PATH) as f:
    _G: nx.DiGraph = nx.node_link_graph(json.loads(f.read()), edges="edges")

# Pre-computed node embeddings — loaded once at import time
# Built by: python -m compressor_crossstorerag.build_node_embeddings
_NODE_EMBEDDINGS_PATH = Path(__file__).parent.parent / "data" / "node_embeddings.json"
with open(_NODE_EMBEDDINGS_PATH) as f:
    _node_embeddings: dict[str, list[float]] = json.load(f)

# Convert to numpy arrays for efficient cosine similarity computation
_node_labels: list[str] = list(_node_embeddings.keys())
_node_matrix: np.ndarray = np.array(list(_node_embeddings.values()))  # shape: (1391, 1536)

# Semantic similarity threshold for node matching
_SEMANTIC_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# PostgreSQL schema for Text-to-SQL prompt
# ---------------------------------------------------------------------------

_PG_SCHEMA = """
Tables in the compressor database:

operational_parameters(parameter TEXT PRIMARY KEY, value TEXT, unit TEXT)
  -- single-value parameters, e.g. maximum_working_pressure, oil_capacity

pressure_settings(ga_model TEXT, icd_model TEXT, dewpoint_variant TEXT,
                  frequency_hz INT, unload_pressure_bar FLOAT,
                  load_pressure_bar FLOAT)
  -- per-model pressure setpoints; ga_model is always 'GA5'

service_plans(plan CHAR(1), interval_hours INT, interval_months INT,
              description TEXT)
  -- service plans A/B/C/D with calendar and hour-based intervals

protection_thresholds(parameter TEXT, level TEXT, value FLOAT, unit TEXT)
  -- shutdown and warning thresholds, e.g. high_temperature shutdown at 120°C
"""

# Sentinel strings that indicate empty or failed retrieval
_EMPTY_SIGNALS = {
    "No matching entities found in knowledge graph.",
    "Entities found but no relationships retrieved.",
    "No relevant passages found.",
    "No matching records found.",
    "SQL error",
}

# Store name mapping for fallback selection
_QUERY_TYPE_TO_STORE = {
    "spec": "postgresql",
    "procedure": "chromadb",
    "relation": "networkx",
    "hybrid": "postgresql",
}


# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------

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
    result = _classifier_llm.invoke(prompt)
    query_type = result.content.strip().lower()
    if query_type not in ("spec", "procedure", "relation", "hybrid"):
        query_type = "procedure"
    return {"query_type": query_type}


# ---------------------------------------------------------------------------
# Node: db_retrieve
# ---------------------------------------------------------------------------

def db_retrieve_node(state) -> dict:
    sql = _generate_sql(state.query)
    rows = _execute_sql(sql)
    accumulated = _append_context(state.accumulated_context, rows)
    return {"sql_context": rows, "accumulated_context": accumulated}


def _generate_sql(query: str) -> str:
    prompt = f"""{_PG_SCHEMA}

Write a single read-only SQL SELECT statement to answer this question.
IMPORTANT: Always use SELECT * — never list specific column names.
Only query for structured data — do not attempt to answer parts of the question
that require component relationships or procedural steps.
Return ONLY the SQL — no explanation, no markdown fences.

Question: {query}
"""
    result = _classifier_llm.invoke(prompt)
    sql = result.content.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    return sql


def _execute_sql(sql: str) -> str:
    try:
        with _pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        if not rows:
            return "No matching records found."
        lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
        return "\n".join(lines)
    except Exception as exc:
        return f"SQL error: {exc}"


# ---------------------------------------------------------------------------
# Node: vector_retrieve
# ---------------------------------------------------------------------------

def vector_retrieve_node(state) -> dict:
    query_vec = _embeddings.embed_query(state.query)
    results = _collection.query(
        query_embeddings=[query_vec],
        n_results=4,
        include=["documents"]
    )
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs) if docs else "No relevant passages found."
    accumulated = _append_context(state.accumulated_context, context)
    return {"vector_context": context, "accumulated_context": accumulated}


# ---------------------------------------------------------------------------
# Node: graph_retrieve
# ---------------------------------------------------------------------------

def graph_retrieve_node(state) -> dict:
    context = _graph_search(state.query)
    accumulated = _append_context(state.accumulated_context, context)
    return {"graph_context": context, "accumulated_context": accumulated}


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _graph_search(query: str) -> str:
    """
    Hybrid node matching: substring union semantic.

    Step 1 — Substring matching (fast, precise for exact phrases)
    Step 2 — Semantic matching (catches related concepts via embeddings)
    Step 3 — Union of both, capped at 5 seed nodes
    Step 4 — Traverse neighbourhood with direction preservation
    """
    query_lower = query.lower()

    # Step 1: Substring matching
    matched_substring = {
        n for n in _G.nodes
        if len(str(n)) > 4 and str(n).lower() in query_lower
    }

    # Step 2: Semantic matching using pre-computed embeddings
    query_vec = np.array(_embeddings.embed_query(query))
    similarities = np.dot(_node_matrix, query_vec) / (
        np.linalg.norm(_node_matrix, axis=1) * np.linalg.norm(query_vec) + 1e-10
    )
    semantic_indices = np.where(similarities >= _SEMANTIC_THRESHOLD)[0]
    # Sort by similarity descending, take top 5
    top_semantic = sorted(semantic_indices, key=lambda i: similarities[i], reverse=True)[:5]
    matched_semantic = {_node_labels[i] for i in top_semantic}

    # Step 3: Union, cap at 5 seed nodes
    matched = list(matched_substring | matched_semantic)[:5]

    if not matched:
        return "No matching entities found in knowledge graph."

    # Step 4: Traverse neighbourhood with direction preservation
    lines: list[str] = []
    for node in matched:
        if node not in _G:
            continue
        successors = list(_G.successors(node))
        predecessors = list(_G.predecessors(node))
        if successors:
            lines.append(f"{node} → {', '.join(str(n) for n in successors[:4])}")
        if predecessors:
            lines.append(f"{', '.join(str(n) for n in predecessors[:4])} → {node}")

    return "\n".join(lines) if lines else "Entities found but no relationships retrieved."


# ---------------------------------------------------------------------------
# Node: hybrid_retrieve
# ---------------------------------------------------------------------------

def _extract_sql_question(query: str) -> str:
    prompt = f"""Extract only the part of this question that can be answered from structured tabular data
(parameters, thresholds, service intervals, pressure settings).
Remove any parts about component relationships, procedures, or steps.
Return only the extracted question, nothing else.

Question: {query}
"""
    result = _classifier_llm.invoke(prompt)
    return result.content.strip()


def hybrid_retrieve_node(state) -> dict:
    sql_query = _extract_sql_question(state.query)
    sql = _generate_sql(sql_query)
    sql_rows = _execute_sql(sql)
    graph_ctx = _graph_search(state.query)
    new_content = f"{sql_rows}\n\n{graph_ctx}"
    accumulated = _append_context(state.accumulated_context, new_content)
    return {
        "sql_context": sql_rows,
        "graph_context": graph_ctx,
        "accumulated_context": accumulated,
    }


# ---------------------------------------------------------------------------
# Helper: append new content to accumulated context
# ---------------------------------------------------------------------------

def _append_context(accumulated: str, new_content: str) -> str:
    """Append new retrieval content to accumulated context."""
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"


# ---------------------------------------------------------------------------
# Node: synthesise
# ---------------------------------------------------------------------------

def _is_useful(value: str) -> bool:
    """Return True if the context value contains meaningful content."""
    if not value:
        return False
    return not any(value.startswith(signal) for signal in _EMPTY_SIGNALS)


def synthesise_node(state) -> dict:
    """
    Assemble context from accumulated_context if available,
    otherwise fall back to individual store fields.
    """
    if state.accumulated_context and _is_useful(state.accumulated_context):
        context = state.accumulated_context
    else:
        parts: list[str] = []
        if _is_useful(state.sql_context):
            parts.append(f"[Structured data]\n{state.sql_context}")
        if _is_useful(state.vector_context):
            parts.append(f"[Narrative/procedural text]\n{state.vector_context}")
        if _is_useful(state.graph_context):
            parts.append(f"[Component relationships]\n{state.graph_context}")
        context = "\n\n---\n\n".join(parts) if parts else "No context retrieved."

    return {"context": context}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def generate_node(state) -> dict:
    prompt = f"""You are a maintenance assistant for an Atlas Copco GA5 air compressor.
Answer the question using only the provided context. Be concise and precise.
If the context does not contain enough information, say so.

Context:
{state.context}

Question: {state.query}
"""
    result = _llm.invoke(prompt)
    previous_answer = state.answer
    answer = result.content.strip()
    return {"answer": answer, "previous_answer": previous_answer}


# ---------------------------------------------------------------------------
# Node: judge
# ---------------------------------------------------------------------------

def judge_node(state) -> dict:
    prompt = f"""You are evaluating whether an answer fully addresses a maintenance question.

Question: {state.query}
Context retrieved: {state.context}
Answer generated: {state.answer}

Evaluate whether the answer:
1. Directly addresses the question asked
2. Contains specific values or steps where the question requires them
3. Is grounded in the provided context — no hallucination

Return a JSON object with exactly three fields:
  "verdict": "good" or "insufficient"
  "score": a float between 0.0 and 1.0 indicating answer completeness
  "evaluation": a one-sentence description of what is missing, or "" if verdict is good

Return ONLY the JSON object — no explanation, no markdown fences.
"""
    result = _classifier_llm.invoke(prompt)

    judge_parse_error = False
    try:
        raw = result.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "insufficient").lower()
        score = float(parsed.get("score", 0.0))
        evaluation = parsed.get("evaluation", "")
        if verdict not in ("good", "insufficient"):
            verdict = "insufficient"
    except Exception:
        judge_parse_error = True
        verdict = "insufficient"
        score = 0.0
        evaluation = "Judge response could not be parsed."

    new_attempt = state.attempt + 1
    new_score_trend = state.score_trend + [score]

    return {
        "verdict": verdict,
        "score": score,
        "evaluation": evaluation,
        "attempt": new_attempt,
        "score_trend": new_score_trend,
        "judge_parse_error": judge_parse_error,
    }


# ---------------------------------------------------------------------------
# Cross-store fallback — LLM selects best store for re-retrieval
# ---------------------------------------------------------------------------

def _select_fallback_store(query_type: str, evaluation: str) -> str:
    """
    Ask the LLM to select the best store for re-retrieval based on
    the judge's evaluation of what is missing.

    Returns one of: "postgresql", "chromadb", "networkx"
    Falls back to the original store if LLM returns unexpected value.
    """
    original_store = _QUERY_TYPE_TO_STORE.get(query_type, "chromadb")

    prompt = f"""An AI assistant tried to answer a maintenance question but the answer was incomplete.

Original query type: {query_type}
Original store used: {original_store}
Judge evaluation — what is missing: {evaluation}

Which store is most likely to contain the missing information?

Stores available:
- postgresql  : exact parameter values, thresholds, service intervals, pressure settings
- chromadb    : narrative procedures, safety instructions, step-by-step text
- networkx    : component relationships, system dependencies, entity connections

Reply with exactly one word: postgresql, chromadb, or networkx.
"""
    result = _classifier_llm.invoke(prompt)
    store = result.content.strip().lower()

    if store not in ("postgresql", "chromadb", "networkx"):
        store = original_store  # safe fallback to original store

    return store


# ---------------------------------------------------------------------------
# Node: re_retrieve — cross-store fallback
# ---------------------------------------------------------------------------

def re_retrieve_node(state) -> dict:
    """
    Targeted re-retrieval with cross-store fallback.

    1. Formulate targeted query from judge's evaluation
    2. Ask LLM which store is most likely to contain missing information
    3. Route to selected store
    4. Append new content to accumulated_context
    """
    # Formulate targeted query from judge's evaluation
    if state.evaluation:
        targeted_query = f"{state.query} — specifically: {state.evaluation}"
    else:
        targeted_query = state.query

    # LLM selects which store to query
    fallback_store = _select_fallback_store(state.query_type, state.evaluation)

    # Route to selected store
    if fallback_store == "postgresql":
        sql = _generate_sql(targeted_query)
        new_content = _execute_sql(sql)

    elif fallback_store == "chromadb":
        query_vec = _embeddings.embed_query(targeted_query)
        results = _collection.query(
            query_embeddings=[query_vec],
            n_results=4,
            include=["documents"]
        )
        docs = results["documents"][0] if results["documents"] else []
        new_content = "\n\n".join(docs) if docs else "No relevant passages found."

    else:  # networkx
        new_content = _graph_search(targeted_query)

    # Append to accumulated context
    accumulated = _append_context(state.accumulated_context, new_content)

    return {
        "accumulated_context": accumulated,
        "fallback_store": fallback_store,
    }
