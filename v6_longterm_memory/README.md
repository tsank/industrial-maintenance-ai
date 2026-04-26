# v6 — Long-term Memory and Cross-Session Persistence

> This repository is part of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.

v6 of the Industrial Maintenance AI architectural progression — Long-term Memory and Cross-Session Persistence over the Atlas Copco GA5 compressor manual.

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
| **v6** | **Long-term Memory — cross-session persistence + episodic memory retrieval** |

---

## Architectural limitation addressed from v5.5

Every invocation in v5.5 starts from scratch. The agent has no memory of previous sessions — a maintenance engineer investigating a recurring fault must repeat the same queries across conversations, and the system cannot draw on prior interactions to improve its answers. Knowledge accumulates in the engineer's head, not in the system.

---

## What was designed and built

Two complementary persistence mechanisms introduced in v6:

**Cross-session state persistence** — LangGraph `PostgresSaver` checkpointing saves and restores the full `AgentState` across process boundaries, keyed by a `session_id`. A maintenance engineer can resume a fault investigation exactly where they left off, days later.

**Episodic memory retrieval** — a queryable store of past interactions (`memory_episodes` table, pgvector similarity search) that the agent draws on at the start of every run. Semantically similar past Q&A pairs are injected as context before retrieval begins, allowing the agent to build on prior answers rather than starting cold.

---

## Architecture

Two new bookend nodes wrap the unchanged v5.5 pipeline:

```
memory_retrieval_node           ← NEW: embed query, fetch top-k past episodes
        │
        ▼
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
                       synthesise_node  ◄───────────────-──────────┐
                       (+ memory context block)                    │
                            │                                      │
                       generate_node                               │
                            │                                      │
                        judge_node                                 │
                            │                                      │
                   should_retry() ── conditional                   │
                            │                                      │
                ┌── retry ──────► re_retrieve_node ────────────────┘
                │
                └── end ──► memory_storage_node   ← NEW: persist episode, update history
                                    │
                                   END
```

The v5.5 pipeline — classify, retrieve, synthesise, generate, judge, re-retrieve — is preserved unchanged in the middle. All new logic is in the two bookend nodes and a single context injection point in `synthesise_node`.

---

## Two distinct memory mechanisms

This is the most important architectural distinction in v6. The two persistence systems serve different purposes and must not be conflated:

| | LangGraph Checkpointing | Episodic Memory |
|---|---|---|
| **What** | Full `AgentState` serialised to binary | Completed Q&A episodes as structured rows |
| **Where** | `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` tables | `memory_episodes` table |
| **Managed by** | `PostgresSaver` (LangGraph internal) | `memory_store.py` (application code) |
| **Keyed by** | `thread_id` (session-scoped) | Semantic similarity (cross-session) |
| **Purpose** | Resume a specific conversation thread | Learn from all past interactions |
| **Restored** | Automatically by LangGraph on next invoke | Queried by `memory_retrieval_node` |

Checkpointing answers: *"Where did this conversation leave off?"*
Episodic memory answers: *"What do I already know about this topic?"*

---

## New in v6

### 1. `memory_retrieval_node` — first node in the graph

Runs before `classify_node` on every invocation. Embeds the incoming query using `text-embedding-3-small`, queries `memory_episodes` for the top-k most semantically similar past episodes via pgvector cosine distance, and writes them to `state.retrieved_memories`.

Running before `classify` means retrieved memories are available on state throughout the entire pipeline — including at synthesis time where they are injected as context.

The query embedding is stored on `AgentState` (`query_embedding` field) and reused by `memory_storage_node` — avoiding a redundant OpenAI API call and guaranteeing embedding consistency within a single run.

---

### 2. `memory_storage_node` — terminal node before END

Runs once per invocation, after the judge has made its final decision. Persists a `MemoryEntry` to `memory_episodes` only if the final score meets the quality threshold (`score >= min_score_threshold`). Low-quality episodes are discarded at write time — the episodic store stays clean by design and retrieval requires no post-hoc filtering.

Records the actual store used for the final answer — `fallback_store` if a cross-store retry occurred, otherwise `query_type`.

---

### 3. Memory context injection in `synthesise_node`

The only change to the v5.5 pipeline. If `retrieved_memories` is non-empty, a labelled memory block is prepended before the retrieval context:

```
[Relevant past interactions]
Q: What is the maximum working pressure of the GA5?
A: The maximum working pressure of the GA5 is 8.5 bar.

---

[Structured data]
ga_model: GA5, unload_pressure_bar: 7.7, load_pressure_bar: 6.7
...
```

Memory block is prepended — not appended — because LLMs weight earlier context more heavily. The label `[Relevant past interactions]` distinguishes past knowledge from fresh retrieval, giving `generate_node` clear signal about the provenance of each context section.

---

### 4. Quality gate

`memory_storage_node` gates persistence on `state.score >= state.min_score_threshold` (default 0.3). This reuses the existing threshold already on `AgentState` — no new constant introduced. Episodes that exit via `max_attempts` exhaustion or declining score trend with a low final score are not persisted. This prevents the agent from learning from its own failures.

---

### 5. pgvector for similarity search

Episode similarity search uses pgvector's `<=>` cosine distance operator directly in SQL:

```sql
SELECT session_id, query, answer, score, store_used, created_at
FROM   memory_episodes
ORDER  BY embedding <=> %s::vector
LIMIT  %s
```

Chosen over the NumPy cosine similarity pattern from v5.5 because the episode store lives in PostgreSQL and SQL-native search avoids loading all embeddings into memory on every query. The `ivfflat` index is omitted at this scale — it requires a minimum row count to build lists and a sequential scan is faster on small tables.

---

## AgentState — new fields

```python
class AgentState(BaseModel):
    # ... all v5.5 fields unchanged ...

    # Long-term memory (v6)
    session_id: str = ""                              # thread_id; generated by run() if empty
    session_history: list[MemoryEntry] = []           # episodes persisted this session
    retrieved_memories: list[MemoryEntry] = []        # top-k past episodes for this query
    memory_injected: bool = False                     # True once memory_retrieval_node ran
    query_embedding: list[float] = []                 # computed once, reused for storage
```

`session_history` is part of the checkpointed state — it accumulates across runs of the same `session_id` and survives process restarts. `retrieved_memories` is populated fresh each run and not meaningful to persist.

---

## `MemoryEntry` — the episode record

```python
class MemoryEntry(BaseModel):
    session_id: str
    query: str
    answer: str
    score: float
    store_used: str
    created_at: datetime
```

`MemoryEntry` never carries an embedding — embeddings are large (1536 floats), only needed for database similarity search, and would bloat the in-memory state unnecessarily. The embedding is passed separately to `write_episode()` and lives only in the database.

---

## Exit conditions — unchanged from v5.5

```
verdict == "good"
    → memory_storage → END

verdict == "insufficient" AND attempt >= max_attempts
    → memory_storage → END

verdict == "insufficient" AND valid JSON AND score < min_score_threshold
    → memory_storage → END

verdict == "insufficient" AND parse error
    → RETRY

verdict == "insufficient" AND valid JSON AND score declining
    → memory_storage → END (with previous_answer)

verdict == "insufficient" AND valid JSON AND score stable/improving
    → RETRY

verdict == "insufficient" AND no trend yet
    → RETRY
```

In all exit paths, `memory_storage_node` runs. Whether it writes an episode depends on the final score.

---

## Known limitations

**Cold start** — on first use the episodic store is empty and `retrieved_memories` is always empty. The memory benefit accrues with use; the system improves over time rather than immediately.

**Duplicate episodes** — repeated test runs write semantically identical episodes to `memory_episodes`. A deduplication strategy (e.g. skip write if identical query exists above threshold) would keep the store clean but involves tradeoffs around query normalisation and threshold sensitivity. This is a deliberate simplification for v6.

**`classify_node` does not use retrieved memories** — memories are on state when `classify_node` runs but are not passed into the classification prompt. Using past episodes as few-shot classification examples would make the retrieval-before-classify ordering architecturally earn its keep. This requires careful prompt design to avoid biasing classification toward historical patterns.

**Three database connections** — `clients.py`, `memory_store.py`, and `agent.py` each hold a separate connection to the same PostgreSQL instance. In production, a shared connection pool would be appropriate.

These limitations involve architectural tradeoffs — deduplication strategy, few-shot prompt design, connection pooling — each of which warrants focused attention. v6's goal is to establish the memory foundation cleanly; refinements build on top of it.

---

## Observability gap — motivating v7

v6 adds memory but has no visibility into what happened during a run. There is no trace of which nodes fired, how long each took, what prompts were sent to the LLMs, or why the judge scored an answer the way it did. Debugging requires adding print statements and re-running. v7 addresses this with LangSmith and/or OpenTelemetry integration.

---

## Setup sequence

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt
python -m pip install "psycopg[binary]"

# 2. Copy and configure environment
cp .env.example .env
# Add OPENAI_API_KEY

# 3. Ensure Docker containers are running
docker ps   # should show chromadb and compressor-pg (ankane/pgvector image)

# 4. Initialise memory schema (one-time)
python scripts/init_db.py

# 5. Run tests
python -m pytest tests/ -v
```

**Note:** `ankane/pgvector` is required as the PostgreSQL image — the standard `postgres:15` image does not include the `vector` extension.

---

## Project structure

```
v6_longterm_memory/
├── compressor_longtermrag/
│   ├── __init__.py
│   ├── agent.py              # StateGraph, PostgresSaver, run() public API
│   ├── state.py              # AgentState + MemoryEntry Pydantic models
│   ├── memory_store.py       # PostgreSQL episode CRUD + pgvector similarity search
│   └── nodes/
│       ├── __init__.py
│       ├── clients.py        # All shared client initialisations (once at import)
│       ├── memory_retrieval.py   # NEW: embed query, fetch top-k episodes
│       ├── memory_storage.py     # NEW: quality gate, persist episode, update history
│       ├── classify.py           # from v5.5, unchanged
│       ├── retrieve_vector.py    # from v5.5, unchanged
│       ├── retrieve_graph.py     # from v5.5, unchanged
│       ├── retrieve_spec.py      # from v5.5, unchanged
│       ├── retrieve_hybrid.py    # from v5.5, unchanged
│       ├── synthesise.py         # from v5.5, + memory context block prepended
│       └── judge.py              # from v5.5, unchanged
├── data/
│   ├── graph.json                # NetworkX graph (copied from v5.5)
│   ├── node_embeddings.json      # Pre-computed node embeddings (copied from v5.5)
│   └── seed/
│       └── init_memory_schema.sql
├── tests/
│   ├── __init__.py
│   ├── test_agent.py             # Standard invoke tests + memory field assertions
│   ├── test_memory_persistence.py # Same thread_id across two invocations
│   └── test_memory_retrieval.py  # Seed episodes, assert similarity retrieval
├── scripts/
│   └── init_db.py                # One-time: enable pgvector, create memory_episodes
├── requirements.txt
├── .env.example
└── README.md
```

---

## Test results

```
6 passed
```

| Test | What it proves |
|---|---|
| `test_spec_query` | Agent answers, `memory_injected=True`, episode stored |
| `test_procedure_query` | Pipeline runs end-to-end, quality gate applied |
| `test_relation_query` | Pipeline runs end-to-end, quality gate applied |
| `test_session_history_entry_fields` | `MemoryEntry` fields fully populated |
| `test_cross_session_persistence` | Same `session_id` across two runs — history accumulates |
| `test_retrieved_memories_populated` | Seeded episode retrieved via pgvector similarity |

---

## Architectural concepts

- LangGraph checkpointing — `PostgresSaver` serialises full `AgentState` across process boundaries
- Episodic memory — structured Q&A history queryable by semantic similarity
- pgvector cosine distance — SQL-native similarity search without loading embeddings into memory
- Quality gate — persist only high-scoring episodes, keep the memory store clean by design
- Embedding reuse — compute once in `memory_retrieval_node`, reuse in `memory_storage_node`
- Bookend pattern — new nodes wrap unchanged pipeline without modifying its internals

## Technical decisions

- `PostgresSaver` requires `psycopg3` (`psycopg[binary]`) — separate from `psycopg2` used by application nodes
- `checkpointer.setup()` is idempotent — safe to call on every module load
- `session_id` on `AgentState` and `thread_id` in LangGraph config are always set to the same value
- `ivfflat` index omitted — requires >30 rows per list to return results; sequential scan correct at this scale
- `MemoryEntry.embedding` excluded from Pydantic model — embeddings passed separately to keep state lightweight
- `min_score_threshold` reused as quality gate — no new constant, consistent with existing judge logic

---

## Data Source

Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03.
Sourced from ManualsLib for educational and research purposes: https://www.manualslib.com/manual/1353128/Atlas-Copco-Ga5.html

---

## Attribution

Built as v6 of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.
Developed with assistance from Claude (Anthropic).
