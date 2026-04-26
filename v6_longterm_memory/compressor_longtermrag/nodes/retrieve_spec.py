"""
retrieve_spec.py — PostgreSQL Text-to-SQL retrieval node. Carried over from v5.5 unchanged.
"""

from __future__ import annotations

import psycopg2.extras

from compressor_longtermrag.nodes.clients import classifier_llm, pg_conn, PG_SCHEMA


def _generate_sql(query: str) -> str:
    prompt = f"""{PG_SCHEMA}

Write a single read-only SQL SELECT statement to answer this question.
IMPORTANT: Always use SELECT * — never list specific column names.
Only query for structured data — do not attempt to answer parts of the question
that require component relationships or procedural steps.
Return ONLY the SQL — no explanation, no markdown fences.

Question: {query}
"""
    result = classifier_llm.invoke(prompt)
    return result.content.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()


def _execute_sql(sql: str) -> str:
    try:
        with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        if not rows:
            return "No matching records found."
        lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
        return "\n".join(lines)
    except Exception as exc:
        return f"SQL error: {exc}"


def db_retrieve_node(state) -> dict:
    sql = _generate_sql(state.query)
    rows = _execute_sql(sql)
    accumulated = _append_context(state.accumulated_context, rows)
    return {"sql_context": rows, "accumulated_context": accumulated}


def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"
