# v7 — Observability

> This repository is part of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.

v7 of the Industrial Maintenance AI architectural progression — Full-stack observability over the Atlas Copco GA5 compressor maintenance assistant, with a Gradio UI separating the end-user chatbot from the operator trace dashboard.

---

## Where this fits in the progression

| Version | Capability |
|---|---|
| v1 | Vector RAG — semantic search over manual |
| v2 | Graph RAG — knowledge graph + vector |
| v2.5 | Semantic chunking — better graph quality |
| v3 | Agentic RAG — classify and route to right retriever |
| v4 | Multi-Store RAG — PostgreSQL, ChromaDB, NetworkX, hybrid paths |
| v5 | Self-Evaluating RAG — judge loop, score trend, adaptive stopping |
| v5.5 | Cross-Store Fallback — smarter graph matching + intelligent retry routing |
| v6 | Long-term Memory — cross-session persistence + episodic memory retrieval |
| **v7** | **Observability — OTel/Jaeger tracing, LangSmith, Gradio UI** |

---

## Architectural limitation addressed from v6

v6 adds memory but has no visibility into what happened during a run. There is no trace of which nodes fired, how long each took, what prompts were sent to the LLMs, or why the judge scored an answer a particular way. Debugging requires adding print statements and re-running. A system that cannot be observed cannot be trusted in production.

---

## What was designed and built

Two complementary observability layers and a production-ready UI:

**OpenTelemetry → Jaeger** — every LangGraph node is instrumented with OTel spans. Node execution time, latency, and node-specific diagnostic attributes (score, verdict, chunks returned, SQL generated) are captured and sent to Jaeger via OTLP. Child spans on helper functions (`_graph_search`, `_generate_sql`, `_execute_sql`) separate LLM latency from database latency within a single node.

**LangSmith** — automatic tracing of every LLM call via LangChain's built-in instrumentation. Full prompt, response, token usage, and per-call latency captured with zero instrumentation code in node files.

**Gradio UI** — a two-tab interface separating the end-user chatbot from the operator observability dashboard. Tab 1 provides a clean conversational interface. Tab 2 shows a Plotly latency chart, a link to the LangSmith project, and a link to Jaeger.

---

## Loom Video

> _To be added after recording_

---

## Architecture

The v6 graph is preserved unchanged. Observability wraps every node via a `@traced_node` decorator factory — node logic is untouched:

```
memory_retrieval_node        ← @traced_node("node.memory_retrieval")
        │
        ▼
classify_node                ← @traced_node("node.classify")
        │
        ▼
route_query() ── conditional
        │
        ├── spec      → db_retrieve_node      ← @traced_node("node.retrieve_spec")
        ├── procedure → vector_retrieve_node  ← @traced_node("node.retrieve_vector")
        ├── relation  → graph_retrieve_node   ← @traced_node("node.retrieve_graph")
        └── hybrid    → hybrid_retrieve_node  ← @traced_node("node.retrieve_hybrid")
                            │
                       synthesise_node        ← @traced_node("node.synthesise")
                            │
                       generate_node          ← @traced_node("node.generate")
                            │
                        judge_node            ← @traced_node("node.judge")
                            │
                   should_retry() ── conditional
                            │
                ┌── retry ──► re_retrieve_node  ← @traced_node("node.re_retrieve")
                │
                └── end ──► memory_storage_node ← @traced_node("node.memory_storage")
                                    │
                                   END
```

---

## The `@traced_node` decorator

The central observability pattern in v7 is a **decorator factory** — a function that returns a decorator, which wraps each node function in an OTel span:

```python
def traced_node(span_name: str):
    def decorator(fn):
        @functools.wraps(fn)          # preserves fn.__name__ for LangGraph
        def wrapper(state):           # LangGraph calls this
            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                try:
                    result = fn(state, span)   # original node receives span
                    return result
                except Exception as e:
                    span.set_status(trace.StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator
```

Applied to every node:

```python
@traced_node("node.classify")
def classify_node(state, span) -> dict:
    ...
    span.set_attribute("query_type", query_type)
    return {"query_type": query_type}
```

**What this achieves:** node logic and observability logic are completely separate. Removing the decorator removes all tracing — the node is untouched. Changing the tracing backend requires editing only `otel.py`.

---

## Child spans on helper functions

For helper functions with rich internal diagnostic data, child spans are opened manually using `get_tracer()`. OTel's context propagation automatically nests them under the parent node span:

```python
def _graph_search(query: str) -> str:
    from compressor_observability.observability import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("helper.graph_search") as child_span:
        ...
        child_span.set_attribute("entities_matched", len(matched))
        child_span.set_attribute("substring_matches", len(matched_substring))
        child_span.set_attribute("semantic_matches", len(matched_semantic))
```

Jaeger displays this as a nested timeline:

```
node.retrieve_graph          450ms
    helper.graph_search      380ms   ← embedding + graph traversal
```

**Why not use the decorator on helpers:** the `@traced_node` decorator assumes `(state, span)` arguments — the LangGraph node contract. Helper functions have different argument signatures. Child spans are the correct OTel pattern for internal function calls.

---

## Observability coverage per node

| Span | Attributes |
|---|---|
| `node.memory_retrieval` | `episodes_retrieved` |
| `node.classify` | `query_type` |
| `node.retrieve_spec` | `sql_generated`, `rows_returned` |
| `helper.generate_sql` | `sql` (truncated to 500 chars) |
| `helper.execute_sql` | `rows_returned`, `success` |
| `node.retrieve_vector` | `chunks_returned`, `context_length` |
| `node.retrieve_graph` | `context_length` |
| `helper.graph_search` | `entities_matched`, `substring_matches`, `semantic_matches` |
| `node.retrieve_hybrid` | `sql_context_length`, `graph_context_length` |
| `node.synthesise` | `memory_injected`, `context_length` |
| `node.generate` | `answer_length` |
| `node.judge` | `score`, `verdict`, `attempt`, `judge_parse_error`, `evaluation` |
| `node.re_retrieve` | `fallback_store`, `new_content_length` |
| `node.memory_storage` | `final_score`, `store_used`, `persisted` |

**Note on `helper.execute_sql`:** `_execute_sql` makes a direct `psycopg2` call — invisible to LangSmith which only intercepts LangChain LLM calls. The child span is the only place PostgreSQL query latency is captured for this operation.

---

## LangSmith vs OTel — complementary, not redundant

| | LangSmith | OTel / Jaeger |
|---|---|---|
| **Activation** | Environment variables — zero instrumentation code | `@traced_node` decorator on each node |
| **Captures** | Every LLM call: prompt, response, token usage, latency | Node execution, latency, custom attributes |
| **Blind to** | Direct DB calls, non-LangChain code | Nothing — explicit spans cover everything |
| **Best for** | Inspecting what the LLM was sent and returned | Debugging node-level performance and failures |
| **Opt-in** | Yes — disabled if `LANGCHAIN_API_KEY` absent | Yes — disabled if `OTLP_ENDPOINT` absent |

---

## Gradio UI

Two tabs serving different audiences:

**Tab 1 — Chatbot** — end-user interface. Clean Q&A over the GA5 manual. No trace information shown. Calls `run()` directly.

**Tab 2 — Observability** — operator/developer interface. Runs a query with timing, displays a Plotly bar chart of total latency, and provides one-click links to LangSmith and Jaeger.

```bash
python -m ui.app
# Open http://localhost:7860
```

---

## New in v7 vs v6

| File | Change |
|---|---|
| `observability/otel.py` | NEW — TracerProvider, OTLP exporter, `traced_node` decorator |
| `observability/langsmith.py` | NEW — LangSmith env configuration, opt-in logic |
| `nodes/clients.py` | +2 lines: `setup_otel()`, `setup_langsmith()` |
| Every node file | +decorator, +`span` argument, +`span.set_attribute()` calls |
| `retrieve_graph.py` | +child span on `_graph_search` |
| `retrieve_spec.py` | +child spans on `_generate_sql`, `_execute_sql` |
| `memory_storage.py` | `datetime.utcnow()` → `datetime.now(timezone.utc)` |
| `ui/` | NEW — Gradio app, chatbot tab, observability tab |
| `docker-compose.yml` | NEW — manages ChromaDB, PostgreSQL, Jaeger together |
| `tests/test_observability.py` | NEW — Jaeger reachability, LangSmith config, OTel pipeline |

---

## Infrastructure

```
Docker containers (managed by docker-compose.yml):
  ChromaDB        chromadb/chroma:0.5.23    port 8000
  PostgreSQL      ankane/pgvector           port 5432
  Jaeger          jaegertracing/all-in-one  port 16686 (UI), 4317 (OTLP)
```

---

## Setup sequence

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Copy and configure environment
cp .env.example .env
# Add OPENAI_API_KEY, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT, OTLP_ENDPOINT

# 3. Start Docker stack
docker-compose up -d

# 4. Initialise memory schema (one-time)
python scripts/init_db.py

# 5. Seed operational tables (one-time, from v4)
# Run compressor_multistorerag.db_setup from v4_multistore_rag/

# 6. Run tests
python -m pytest tests/ -v

# 7. Launch UI
python -m ui.app
```

**Note:** `data/graph.json` and `data/node_embeddings.json` are excluded from version control (large files). Copy from v6 or regenerate from v2/v2.5.

---

## Project structure

```
v7_observability/
├── compressor_observability/
│   ├── __init__.py                  # run() public API
│   ├── agent.py                     # StateGraph, PostgresSaver, run()
│   ├── state.py                     # AgentState + MemoryEntry Pydantic models
│   ├── memory_store.py              # PostgreSQL episode CRUD + pgvector search
│   ├── observability/               # NEW
│   │   ├── otel.py                  # TracerProvider, OTLP exporter, traced_node
│   │   └── langsmith.py             # LangSmith env config, opt-in activation
│   └── nodes/
│       ├── clients.py               # +setup_otel(), +setup_langsmith()
│       ├── memory_retrieval.py      # +@traced_node
│       ├── memory_storage.py        # +@traced_node, datetime fix
│       ├── classify.py              # +@traced_node
│       ├── retrieve_vector.py       # +@traced_node
│       ├── retrieve_graph.py        # +@traced_node, +child span on _graph_search
│       ├── retrieve_spec.py         # +@traced_node, +child spans on helpers
│       ├── retrieve_hybrid.py       # +@traced_node
│       ├── synthesise.py            # +@traced_node
│       └── judge.py                 # +@traced_node
├── ui/                              # NEW
│   ├── app.py                       # Gradio entry point
│   ├── chatbot_tab.py               # Tab 1: end-user Q&A
│   └── observability_tab.py         # Tab 2: latency chart, backend links
├── data/
│   └── seed/
│       ├── init_memory_schema.sql
│       ├── operational_parameters.csv
│       ├── pressure_settings.csv
│       ├── service_plans.csv
│       └── protection_thresholds.csv
├── tests/
│   ├── test_agent.py
│   ├── test_memory_persistence.py
│   ├── test_memory_retrieval.py
│   └── test_observability.py        # NEW
├── scripts/
│   └── init_db.py
├── docker-compose.yml               # NEW
├── requirements.txt
├── .env.example
└── README.md
```

---

## Test results

```
9 passed
```

| Test | What it proves |
|---|---|
| `test_spec_query` | Agent answers, `memory_injected=True`, episode stored |
| `test_procedure_query` | Pipeline runs end-to-end, quality gate applied |
| `test_relation_query` | Pipeline runs end-to-end, quality gate applied |
| `test_session_history_entry_fields` | `MemoryEntry` fields fully populated |
| `test_cross_session_persistence` | Same `session_id` across two runs — history accumulates |
| `test_retrieved_memories_populated` | Seeded episode retrieved via pgvector similarity |
| `test_langsmith_env_configured` | LangSmith credentials present and correctly set |
| `test_jaeger_ui_reachable` | Jaeger responds at localhost:16686 |
| `test_otel_trace_reaches_jaeger` | Full OTel → Jaeger pipeline verified end-to-end |

---

## Architectural concepts

- OTel decorator factory — `traced_node(span_name)` wraps node functions without touching their logic
- Child spans — helper functions instrument themselves via `get_tracer()`, nested automatically by OTel context propagation
- LangSmith auto-instrumentation — zero code required in nodes; activated by environment variables
- Separation of concerns — observability layer (`observability/`) is completely independent of node logic
- Opt-in observability — both LangSmith and Jaeger gracefully no-op when credentials are absent

## Technical decisions

- `@functools.wraps(fn)` on `wrapper` preserves `fn.__name__` — LangGraph uses function names during graph construction
- `BatchSpanProcessor` — spans buffered and sent in batches rather than per-span network calls
- `insecure=True` on `OTLPSpanExporter` — no TLS required for local Jaeger; production deployments should enable TLS
- `get_tracer()` imported lazily inside helper functions — avoids circular import risk at module load time
- `sql[:500]` truncation on `helper.generate_sql` span — OTel attribute size limits; 500 chars captures all realistic SQL
- `datetime.now(timezone.utc)` replaces deprecated `datetime.utcnow()` — timezone-aware datetime throughout

---

## Limitation addressed in this version

v6 limitation resolved: every run is now fully traceable. Which nodes fired, how long each took, what the LLM received and returned, and why the judge scored an answer a particular way — all visible in Jaeger and LangSmith without adding a single print statement.

## Limitation motivating the next version

The evaluation framework is manual — a human must inspect LangSmith and Jaeger traces to assess answer quality. There is no automated evaluation suite that runs a fixed question set, scores answers against ground truth, and tracks quality over time. v8 will introduce systematic evaluation.

---

## Data Source

Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03.
Sourced from ManualsLib for educational and research purposes: https://www.manualslib.com/manual/1353128/Atlas-Copco-Ga5.html

---

## Attribution

Built as v7 of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.
Developed with assistance from Claude (Anthropic).
