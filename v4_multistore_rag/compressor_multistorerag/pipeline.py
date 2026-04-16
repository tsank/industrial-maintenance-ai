"""
pipeline.py — Node implementations for the multi-store agent.

All expensive client objects (DB connection, ChromaDB, NetworkX graph, LLM)
are initialised once at module import time.  Node functions are plain
callables that accept and return AgentState dicts.
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

# PostgreSQL
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

# ---------------------------------------------------------------------------
# PostgreSQL schema (used in Text-to-SQL prompt)
# ---------------------------------------------------------------------------

_PG_SCHEMA = """
Tables in the compressor database:

operational_parameters(parameter TEXT PRIMARY KEY, value TEXT, unit TEXT)
  -- single-value parameters, e.g. maximum_working_pressure, oil_capacity

pressure_settings(ga_model TEXT, icd_model TEXT, dewpoint_variant TEXT,
                  frequency_hz INT, unload_pressure_bar FLOAT,
                  load_pressure_bar FLOAT, PRIMARY KEY(ga_model, icd_model,
                  dewpoint_variant, frequency_hz))
  -- per-model pressure setpoints; ga_model is always 'GA5'

service_plans(plan CHAR(1), interval_hours INT, interval_months INT,
              description TEXT, PRIMARY KEY(plan))
  -- service plans A/B/C/D with calendar and hour-based intervals

protection_thresholds(parameter TEXT, level TEXT, value FLOAT, unit TEXT,
                      PRIMARY KEY(parameter, level))
  -- shutdown and warning thresholds, e.g. high_temperature shutdown at 120 °C
"""


# ---------------------------------------------------------------------------
# Node: classify
# ---------------------------------------------------------------------------

def classify_node(state: dict) -> dict:
    """Classify query into spec | procedure | relation | hybrid."""
    prompt = f"""You are a query router for an air compressor maintenance assistant.

Classify the following query into exactly one of these types:
- spec        : asks for a specific parameter value, threshold, setting, or service interval
                that lives in a structured table (pressure, temperature, oil, service plan)
- procedure   : asks for steps, instructions, or narrative guidance (how to do something)
- relation    : asks about relationships between components, what connects to what, or
                dependencies in the system
- hybrid      : requires BOTH a structured value AND either component relationships OR
                procedural narrative to answer fully

Reply with a single word — the query type.

Query: {state["query"]}
"""
    result = _classifier_llm.invoke(prompt)
    query_type = result.content.strip().lower()
    if query_type not in ("spec", "procedure", "relation", "hybrid"):
        query_type = "procedure"  # safe default
    return {**state, "query_type": query_type}


# ---------------------------------------------------------------------------
# Node: db_retrieve  (PostgreSQL Text-to-SQL)
# ---------------------------------------------------------------------------

def db_retrieve_node(state: dict) -> dict:
    sql = _generate_sql(state["query"])
    print(f"\n[SQL GENERATED]\n{sql}")
    rows = _execute_sql(sql)
    print(f"\n[SQL RESULT]\n{rows}")
    return {**state, "sql_context": rows}


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
    # strip accidental fences
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
# Node: vector_retrieve  (ChromaDB, optional metadata filter)
# ---------------------------------------------------------------------------

def vector_retrieve_node(state: dict) -> dict:
    query = state["query"]
    query_vec = _embeddings.embed_query(query)

    # apply metadata filter based on query type hint
    # where = _infer_chroma_filter(query)
    # kwargs: dict = {"query_embeddings": [query_vec], "n_results": 4, "include": ["documents"]}
    # if where:
    #     kwargs["where"] = where

    results = _collection.query(
        query_embeddings=[query_vec],
        n_results=4,
        include=["documents"]
    )
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs) if docs else "No relevant passages found."
    return {**state, "vector_context": context}

# ---------------------------------------------------------------------------
# Node: graph_retrieve  (NetworkX neighbourhood traversal)
# ---------------------------------------------------------------------------

def graph_retrieve_node(state: dict) -> dict:
    context = _graph_search(state["query"])
    return {**state, "graph_context": context}


def _graph_search(query: str) -> str:
    """Find nodes whose labels overlap with query tokens; return neighbours."""
    # tokens = {t.lower() for t in query.split() if len(t) > 3}
    # matched = [
    #     n for n in _G.nodes
    #     if any(tok in str(n).lower() for tok in tokens)
    # ]
    query_lower = query.lower()
    matched = [n for n in _G.nodes if len(str(n)) > 4 and str(n).lower() in query_lower]
    if not matched:
        return "No matching entities found in knowledge graph."

    # lines: list[str] = []
    # for node in matched[:5]:  # limit to 5 seed nodes
    #     neighbours = list(_G.successors(node)) + list(_G.predecessors(node))
    #     if neighbours:
    #         lines.append(f"{node} → {', '.join(str(n) for n in neighbours[:8])}")

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
# Node: hybrid_retrieve  (PostgreSQL + NetworkX)
# ---------------------------------------------------------------------------

def hybrid_retrieve_node(state: dict) -> dict:
    # For SQL, only ask about the structured part of the query
    sql_query = _extract_sql_question(state["query"])
    sql = _generate_sql(sql_query)
    print(f"\n[HYBRID SQL GENERATED]\n{sql}")
    sql_rows = _execute_sql(sql)
    graph_ctx = _graph_search(state["query"])
    return {**state, "sql_context": sql_rows, "graph_context": graph_ctx}

def _extract_sql_question(query: str) -> str:
    """Ask the LLM to extract only the structured/tabular part of the query."""
    prompt = f"""Extract only the part of this question that can be answered from structured tabular data
(parameters, thresholds, service intervals, pressure settings).
Remove any parts about component relationships, procedures, or steps.
Return only the extracted question, nothing else.

Question: {query}
"""
    result = _classifier_llm.invoke(prompt)
    return result.content.strip()


# ---------------------------------------------------------------------------
# Node: synthesise  (merge populated context fields → context)
# ---------------------------------------------------------------------------

_EMPTY_SIGNALS = {
    "No matching entities found in knowledge graph.",
    "Entities found but no relationships retrieved.",
    "No relevant passages found.",
    "No matching records found.",
    "SQL error",
}


def synthesise_node(state: dict) -> dict:
    parts: list[str] = []

    def _is_useful(value: str) -> bool:
        """Return True if the context value contains meaningful content."""
        if not value:
            return False
        return not any(value.startswith(signal) for signal in _EMPTY_SIGNALS)

    if _is_useful(state.get("sql_context", "")):
        parts.append(f"[Structured data]\n{state['sql_context']}")
    if _is_useful(state.get("vector_context", "")):
        parts.append(f"[Narrative/procedural text]\n{state['vector_context']}")
    if _is_useful(state.get("graph_context", "")):
        parts.append(f"[Component relationships]\n{state['graph_context']}")

    context = "\n\n---\n\n".join(parts) if parts else "No context retrieved."
    return {**state, "context": context}


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def generate_node(state: dict) -> dict:
    print(f"\n[GENERATE] context: '{state.get('context')}'")
    prompt = f"""You are a maintenance assistant for an Atlas Copco GA5 air compressor.
Answer the question using only the provided context.  Be concise and precise.
If the context does not contain enough information, say so.

Context:
{state['context']}

Question: {state['query']}
"""
    result = _llm.invoke(prompt)
    return {**state, "answer": result.content.strip()}
