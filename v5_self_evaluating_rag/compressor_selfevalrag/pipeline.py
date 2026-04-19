"""
pipeline.py — Node implementations for the self-evaluating agent.

Carried forward from v4:
  classify_node, db_retrieve_node, vector_retrieve_node,
  graph_retrieve_node, hybrid_retrieve_node, synthesise_node, generate_node

New in v5:
  judge_node        — evaluates answer quality, scores it, updates score_trend
  re_retrieve_node  — targeted re-retrieval guided by judge's evaluation

Key change from v4:
  All nodes use Pydantic state — state.field instead of state["field"]
  Return pattern: state.model_copy(update={...}) instead of {**state, "field": value}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import networkx as nx
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

_GRAPH_PATH = Path(__file__).parent.parent / "data" / "graph.json"
with open(_GRAPH_PATH) as f:
    _G: nx.DiGraph = nx.node_link_graph(json.loads(f.read()), edges="edges")

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


def _graph_search(query: str) -> str:
    query_lower = query.lower()
    matched = [
        n for n in _G.nodes
        if len(str(n)) > 4 and str(n).lower() in query_lower
    ]
    if not matched:
        return "No matching entities found in knowledge graph."

    lines: list[str] = []
    for node in matched[:5]:
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
    previous_answer = state.answer   # save current answer before overwriting
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

    # Parse judge response
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

    # Update attempt counter and score trend
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
# Node: re_retrieve
# ---------------------------------------------------------------------------

def re_retrieve_node(state) -> dict:
    """
    Targeted re-retrieval guided by judge's evaluation of what is missing.
    Uses the judge's evaluation to formulate a focused follow-up query.
    Appends new results to accumulated_context rather than replacing it.
    """
    # Formulate targeted query from judge's evaluation
    if state.evaluation:
        targeted_query = f"{state.query} — specifically: {state.evaluation}"
    else:
        targeted_query = state.query

    # Determine which store to re-query based on query_type
    if state.query_type == "spec":
        sql = _generate_sql(targeted_query)
        new_content = _execute_sql(sql)

    elif state.query_type == "procedure":
        query_vec = _embeddings.embed_query(targeted_query)
        results = _collection.query(
            query_embeddings=[query_vec],
            n_results=4,
            include=["documents"]
        )
        docs = results["documents"][0] if results["documents"] else []
        new_content = "\n\n".join(docs) if docs else "No relevant passages found."

    elif state.query_type == "relation":
        new_content = _graph_search(targeted_query)

    else:  # hybrid
        sql_query = _extract_sql_question(targeted_query)
        sql_rows = _execute_sql(_generate_sql(sql_query))
        graph_ctx = _graph_search(targeted_query)
        new_content = f"{sql_rows}\n\n{graph_ctx}"

    # Append to accumulated context
    accumulated = _append_context(state.accumulated_context, new_content)

    return {"accumulated_context": accumulated}
