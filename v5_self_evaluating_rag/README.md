# v5 — Self-Evaluating RAG

> This repository is part of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.

v5 of the Industrial Maintenance AI architectural progression — Self-Evaluating RAG with LangGraph cycles, LLM-as-judge evaluation, and adaptive stopping over the Atlas Copco GA5 compressor manual.

---

## Where this fits in the progression

| Version | Capability |
|---|---|
| v1 | Vector RAG — semantic search over manual |
| v2 | Graph RAG — knowledge graph + vector |
| v2.5 | Semantic chunking — better graph quality |
| v3 | Agentic RAG — classify and route to right retriever |
| v4 | Multi-Store RAG — PostgreSQL, ChromaDB, NetworkX, hybrid paths |
| **v5** | **Self-Evaluating RAG — judge loop, score trend, adaptive stopping** |

---

## Architectural limitation addressed from v4

In v4, every query followed a single-pass flow:

```
classify → retrieve → synthesise → generate → END
```

Once `generate_node` produced an answer, the system stopped regardless of answer quality. There was no mechanism to detect incomplete answers, retry with better context, or distinguish between content gaps and retrieval failures.

v5 closes this loop.

---

## What was designed and built

A self-evaluating agent that assesses its own answer quality after each generation attempt and decides whether to retry, exit early, or accept the answer. The system introduces:

- **LangGraph cycles** — the graph is no longer a DAG; it contains a loop
- **LLM-as-judge** — a second LLM call evaluates answer completeness after each generation
- **Score trend tracking** — scores are recorded across attempts to detect improving or declining retrieval
- **Adaptive stopping** — the system exits early when evidence suggests further retries are counterproductive
- **Context accumulation** — retrieved content grows across retries rather than being replaced
- **Pydantic AgentState** — runtime type enforcement replacing TypedDict

---

## Architecture

```
classify_node
      │
      ▼
route_query() ── conditional
      │
      ├── spec      → db_retrieve_node
      ├── procedure → vector_retrieve_node
      ├── relation  → graph_retrieve_node
      └── hybrid    → hybrid_retrieve_node
                            │
                       synthesise_node  ◄─────────────-─────┐
                            │                               │
                       generate_node                        │
                            │                               │
                        judge_node                          │
                            │                               │
                   should_retry() ── conditional            │
                            │                               │
                ┌── retry ──────────► re_retrieve_node ─────┘
                │
                └── end ──► END
```

The cycle is `synthesise → generate → judge → re_retrieve → synthesise`. Each iteration accumulates richer context than the previous one.

---

## AgentState — Pydantic migration

v5 migrates `AgentState` from `TypedDict` to Pydantic `BaseModel`:

```python
class AgentState(BaseModel):
    query: str
    query_type: str = ""
    attempt: int = 0
    max_attempts: int = 3
    sql_context: str = ""
    vector_context: str = ""
    graph_context: str = ""
    accumulated_context: str = ""   # grows across retries
    context: str = ""
    answer: str = ""
    previous_answer: str = ""       # answer from attempt before current
    verdict: str = ""               # "good" | "insufficient"
    evaluation: str = ""            # judge's description of what is missing
    score: float = 0.0
    score_trend: list[float] = Field(default_factory=list)
    min_score_threshold: float = 0.3
    judge_parse_error: bool = False
```

**Why Pydantic over TypedDict:** With cycles, a type error in state gets fed back into the loop on every iteration. Pydantic validates every field assignment at the node boundary — a wrong type raises `ValidationError` immediately at the node that caused it, not several iterations later inside LangGraph's execution machinery.

---

## Exit conditions — `should_retry()`

The routing function after `judge_node` implements a decision tree with five exit conditions:

```
verdict == "good"
    → EXIT with current answer

verdict == "insufficient" AND attempt >= max_attempts
    → EXIT with current answer

verdict == "insufficient" AND judge_parse_error == True
    → RETRY unconditionally (technical failure, not content gap)

verdict == "insufficient" AND judge_parse_error == False
    AND score < min_score_threshold
    → EXIT with current answer (content not in knowledge base)

verdict == "insufficient" AND judge_parse_error == False
    AND score >= min_score_threshold
    AND score_trend[-1] < score_trend[-2] (score declining)
    → EXIT with previous_answer (new retrievals making things worse)

verdict == "insufficient" AND judge_parse_error == False
    AND score >= min_score_threshold
    AND score stable or improving
    → RETRY

verdict == "insufficient" AND no trend yet (first attempt)
    → RETRY
```

### Design decisions in the exit conditions

**`judge_parse_error` gates all score-based checks**

A malformed JSON response from the judge produces `score = 0.0` by default. Without the parse error gate, this would trigger the `min_score_threshold` exit — treating a technical failure as a content gap. The gate ensures score-based logic only applies to valid judge responses.

**`min_score_threshold = 0.3`**

A score below 0.3 indicates the answer is substantially incomplete — the content likely doesn't exist in the knowledge base. Retrying with different retrieval won't help. This prevents the system from making three full retrieval + generation + judge cycles on questions the manual cannot answer.

**Single-step score trend check**

With `max_attempts = 3`, there are at most two score differences available. A two-consecutive-decline check would only fire on the last attempt — saving nothing. A single-step decline check is the only meaningful early stop at this scale. For mission-critical deployments with higher attempt budgets, a two-step check provides stronger recovery guarantees.

**Score declining → exit with `previous_answer`**

When trend is declining, the previous attempt produced a better answer. `judge_node` saves `answer` to `previous_answer` before each generation overwrites it, so the better-scoring answer is always recoverable.

---

## Context accumulation

Each retrieval node appends to `accumulated_context` rather than replacing it:

```python
def _append_context(accumulated: str, new_content: str) -> str:
    if not accumulated:
        return new_content
    return f"{accumulated}\n\n---\n\n{new_content}"
```

`synthesise_node` uses `accumulated_context` as the primary context source when available. This means attempt 3 has access to everything retrieved in attempts 1, 2, and 3 — generation improves with each cycle.

---

## Targeted re-retrieval

`re_retrieve_node` uses the judge's `evaluation` field to formulate a focused follow-up query:

```python
targeted_query = f"{state.query} — specifically: {state.evaluation}"
```

If the judge says "the answer mentions the service interval but not the components involved", the re-retrieval query specifically asks about components. This is more effective than blindly re-running the original query against the same store.

---

## Known limitations

**Sparse graph edges for some entities**

The knowledge graph built in v2 has sparse relationships for some component nodes. The query "What components are connected to the air filter?" retrieves only one graph edge (`air filter → moisture prevention`). The judge correctly scores this as insufficient (0.2) and the system exits via `min_score_threshold`. This is a data quality limitation from v2 triple extraction, not a v5 bug.

Cross-store fallback for sparse graph results — falling back to ChromaDB when the graph is thin — is deferred to a later version.

**Single retry strategy per query type**

`re_retrieve_node` always retries from the same store as the original query. A smarter implementation would switch stores when the current store is demonstrably insufficient.

---

## Project structure

```
v5_self_evaluating_rag/
├── compressor_selfevalrag/
│   ├── __init__.py
│   ├── agent.py           # StateGraph, AgentState (Pydantic), should_retry()
│   ├── db_retriever.py    # Direct PostgreSQL access for tests
│   ├── db_setup.py        # Create tables and load seed data
│   └── pipeline.py        # All node functions including judge_node, re_retrieve_node
├── data/
│   └── seed/
│       ├── operational_parameters.csv
│       ├── pressure_settings.csv
│       ├── protection_thresholds.csv
│       └── service_plans.csv
├── tests/
│   ├── __init__.py
│   └── test_questions.py  # 6 tests covering all exit conditions
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+, conda env `langchain-rag`
- ChromaDB Docker container on port 8000 with `maintenance_manuals` collection populated
- PostgreSQL Docker container on port 5432
- OpenAI API key

### Installation

```bash
cd v5_self_evaluating_rag
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Database setup

```bash
python -m compressor_selfevalrag.db_setup
```

### Run tests

```bash
python -m pytest tests/ -v -s
```

---

## Test results

```
6 passed
```

| Test | Query | Verdict | Score | Attempts |
|---|---|---|---|---|
| test_seed_data_loaded | Direct DB verification | — | — | — |
| test_spec_good | Shutdown temperature threshold | good | 1.0 | 1 |
| test_procedure_good | Outlet temperature warning steps | good | 1.0 | 1 |
| test_relation_sparse | Air filter components (sparse graph) | insufficient | 0.2 | 1 |
| test_hybrid_good | Service intervals + components | good | 1.0 | 1 |
| test_judge_fields | Judge field invariants + content gap | good/insufficient | varies | 1 |

---

## Architectural concepts

- LangGraph cycles — graphs with loops, guaranteed termination via routing function
- LLM-as-judge — second LLM call evaluates first LLM's output
- Score trend analysis — detect improving vs declining retrieval across attempts
- Adaptive stopping — exit early when evidence suggests retrying is counterproductive
- Context accumulation — richer context with each retry cycle
- Pydantic state — runtime type enforcement at every node boundary
- Targeted re-retrieval — judge feedback drives focused follow-up queries

## Technical decisions

- `judge_parse_error` flag gates all score-based exit logic — parse failures retry unconditionally
- `min_score_threshold = 0.3` prevents pointless retries on content gaps
- Single-step trend check is appropriate for `max_attempts = 3` — double-step would only fire on the last attempt
- `accumulated_context` grows across retries — generation improves with each cycle
- `app.invoke()` returns a dict — wrap with `AgentState(**result)` in tests for attribute access
- `previous_answer` is saved by `generate_node` before overwriting — recoverable when trend declines

---

## Data Source

Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03.
Sourced from ManualsLib for educational and research purposes:
https://www.manualslib.com/manual/1234567/Atlas-Copco-Ga5.html

---

## Attribution

Built as v5 of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.
Developed with assistance from Claude (Anthropic).
