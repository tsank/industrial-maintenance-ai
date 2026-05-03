"""
retrieve_spec.py — PostgreSQL Text-to-SQL retrieval node. Following changes are required from v6:
1. Modifying import statement to import from `compressor_observability.nodes.clients' 
   and compressor_observability.observability
2. Add decorator @traced_node("node.retrieve_spec") to `db_retrieve_node` function
3. Pass two arguments to `db_retrieve_node(state, span) instead of passing only `state`
4. Set `span.set_attribute("sql_generated", sql) and set `span.set_attribute("rows_returned", len(rows.splitlines()))
5. We add child spans to two helper functions:
    - "helper.generate_sql" for `_generate_sql()` to trace the total time of SQL generation
       including the LLM call — LangSmith captures the LLM portion, child span captures the full helper latency
    - "helper.execute_sql" for `_execute_sql()` to trace the PostgreSQL query execution
    — this is not captured by LangSmith since it is a direct DB call, not an LLM call
    - The tracing comments added based on differnet scenarios based on the success of SQL operations
"""

from __future__ import annotations

import psycopg2.extras

from compressor_observability.nodes.clients import classifier_llm, pg_conn, PG_SCHEMA
from compressor_observability.observability import traced_node

def _generate_sql(query: str) -> str:
    from compressor_observability.observability import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("helper.generate_sql") as child_span:
        prompt = f"""{PG_SCHEMA}

Write a single read-only SQL SELECT statement to answer this question.
IMPORTANT: Always use SELECT * — never list specific column names.
Only query for structured data — do not attempt to answer parts of the question
that require component relationships or procedural steps.
Return ONLY the SQL — no explanation, no markdown fences.

Question: {query}
"""
        result = classifier_llm.invoke(prompt)
        sql = result.content.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        child_span.set_attribute("sql", sql[:500])  # Jaeger attribute limit of 4K
        return sql

def _execute_sql(sql: str) -> str:
    from compressor_observability.observability import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("helper.execute_sql") as child_span:
        try:
            with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchall()
            if not rows:
                child_span.set_attribute("rows_returned", 0)
                child_span.set_attribute("success", True)
                return "No matching records found."
            lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
            child_span.set_attribute("rows_returned", len(rows))
            child_span.set_attribute("success", True)
            return "\n".join(lines)
        except Exception as exc:
            child_span.set_attribute("rows_returned", 0)
            child_span.set_attribute("success", False)
            return f"SQL error: {exc}"

@traced_node("node.retrieve_spec")
def db_retrieve_node(state, span) -> dict:
    sql = _generate_sql(state.query)
    rows = _execute_sql(sql)
    accumulated = _append_context(state.accumulated_context, rows)
    span.set_attribute("sql_generated", sql)
    span.set_attribute("rows_returned", len(rows.splitlines()))
    return {"sql_context": rows, "accumulated_context": accumulated}


def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"
