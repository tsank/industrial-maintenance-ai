# Industrial Maintenance AI

A structured architectural progression through increasingly sophisticated
Retrieval-Augmented Generation (RAG) systems, grounded in a real industrial use case —
the Atlas Copco GA5 air compressor maintenance manual.

Each version in this series answers one question: **why wasn't the previous solution enough?**
Every architectural decision is justified by a concrete limitation of the prior system.
This is a record of progressive architectural thinking.

---

## The Use Case

An AI assistant for air compressor maintenance that answers questions from the Atlas Copco GA5
Instruction Book (Document No. 2920 1461 03). The manual contains four types of knowledge:

| Knowledge type | Example question | Best store |
|---|---|---|
| Exact structured facts | What is the shutdown temperature threshold? | PostgreSQL |
| Narrative / procedural text | What should I do when the temperature warning appears? | ChromaDB (vector) |
| Component relationships | What components are connected to the air filter? | NetworkX (graph) |
| Compound questions | What are the service intervals and which components are involved? | PostgreSQL + NetworkX |
| Incomplete or uncertain answers | Any of the above where retrieval was insufficient | LLM-as-judge + retry |

The progression builds toward a system that can answer all four types correctly —
routing each question to the store or stores architecturally suited to answer it.

---

## Progression Overview

```
v1  — Vector RAG
    |
    └── v2  — Graph RAG
            |
            └── v2.5 — Semantic Chunking
                    |
                    └── v3  — Agentic RAG
                            |
                            └── v4  — Multi-Store Agentic RAG
                                    |
                                    └── v5  — Self-Evaluating RAG
                                            |
                                            └── v5.5 — Cross-Store Fallback and Smarter Graph Retrieval
                                                    |
                                                    └── v6  — Long-term Memory and Cross-Session Persistence
                                                            |
                                                            └── v7  — Observability, Tracing, and Gradio UI
```

Each version adds exactly one new architectural concept in response to a real limitation
of the previous version.

---

## v1 — Vector RAG

**Repo:** [`maintenance-rag`](https://github.com/tsank/maintenance-rag)
**Directory:** [`v1_vector_rag/`](v1_vector_rag/)

### What was designed and built
A document ingestion pipeline and vector retrieval system. The GA5 manual was extracted
from PDF, chunked into fixed-size pieces, embedded using `text-embedding-3-small`, and
stored in ChromaDB. Queries were matched against stored chunks by cosine similarity.

### Architectural concepts
- Semantic similarity search — matching meaning, not keywords
- PDF extraction cascade — PyMuPDF → pdfplumber → pytesseract OCR
- Garbled text detection — `is_garbled()`, `(cid:)` detection, `is_menu_diagram()` filters
- ChromaDB as a vector store — HTTP client, collections, embeddings

### Technical decisions
- The embedding model used at ingest time must match the model used at query time
- PDF extraction quality varies by page — a cascade of strategies handles different failure modes
- Fixed-size chunking splits semantic units arbitrarily — a chunk boundary mid-procedure
  produces two incomplete, poorly retrievable chunks

### Limitation that motivated the next version
Vector search retrieves semantically similar text but has no concept of relationships
between entities. "What components connect to the air filter?" returns chunks that mention
air filters — not a structured answer about which components are actually connected.
The manual's knowledge exists as isolated passages, not as a connected network of facts.

---

## v2 — Graph RAG

**Repo:** [`compressor-graphrag`](https://github.com/tsank/compressor-graphrag)
**Directory:** [`v2_graph_rag/`](v2_graph_rag/)

### What was designed and built
A knowledge graph constructed from the manual using SpaCy NER for entity recognition
and `GPT-4o-mini` for triple extraction. Each triple — subject, predicate, object — became
a directed edge in a NetworkX DiGraph. A LangGraph 2-node pipeline combined vector
retrieval with graph traversal.

### Architectural concepts
- Knowledge graphs as a retrieval store — relationships as first-class data structures
- Triple extraction — LLM-based (subject, predicate, object) extraction from text
- NetworkX DiGraph — directed graph with successor/predecessor traversal
- LangGraph StateGraph — nodes, edges, TypedDict state, `{**state, "field": value}` pattern

### Technical decisions
- `nx.node_link_graph(json.loads(...), edges="edges")` required for NetworkX 3.x
- Triple extraction quality depends on input quality — mixed-content pages produce noisy triples
- Nodes do work; routing functions navigate — keep them strictly separate

### Limitation that motivated the next version
Triple extraction operated on whole pages — semantically mixed content. A page about oil
maintenance might contain specifications, warnings, and procedures — the extractor
produced a jumbled mix of triples. Better chunking was needed before extraction.

---

## v2.5 — Semantic Chunking

**Directory:** [`v2_graph_rag/`](v2_graph_rag/) (added to shared `pdf_loader.py`)

### What was designed and built
A semantic chunking function added to the shared PDF loader. Rather than splitting at
fixed character counts, chunks were split at points of low semantic similarity between
adjacent sentences — identified by cosine similarity between sentence embeddings,
with splits at the 25th percentile of similarity scores.

### Architectural concepts
- Chunking as a semantic operation — split at topic boundaries, not character counts
- Cosine similarity between adjacent sentence embeddings as a split signal
- Percentile-based threshold — adaptive to document content

### Technical decisions
- Chunking strategy affects every downstream component — not just vector retrieval,
  but also graph construction quality
- The graph grew from 1004 nodes / 387 edges to 1391 nodes / 686 edges after
  switching to semantic chunking

### Limitation that motivated the next version
Even with better chunking, the system used a fixed retrieval strategy — always vector
search, or always vector + graph. Different questions need different strategies.
"What is the shutdown temperature?" doesn't benefit from graph traversal.
"What components connect to the air filter?" doesn't benefit from vector search.

---

## v3 — Agentic RAG

**Repo:** [`compressor-agenticrag`](https://github.com/tsank/compressor-agenticrag)
**Directory:** [`v3_agentic_rag/`](v3_agentic_rag/)

### What was designed and built
A LangGraph StateGraph with a classification node that routes queries to one of three
specialised retrieval nodes — vector retrieval, graph retrieval, or spec retrieval —
based on query type. Conditional edges replaced the linear pipeline.

### Architectural concepts
- Query classification — identify what kind of question before deciding how to answer it
- Conditional routing — `add_conditional_edges(source, routing_fn, mapping)`
- Agentic architecture — the system makes a decision before taking action
- Spec retrieval — token overlap matching against extracted specifications JSON

### Technical decisions
- `add_conditional_edges` takes a function reference, not a function call
- `StateGraph(AgentState)` takes the TypedDict class as a blueprint, not an instance
- Module-level client initialisation — expensive operations happen once at import time

### Limitation that motivated the next version
Spec retrieval used a JSON file populated by LLM extraction — probabilistic and
unverified for safety-critical values. More critically, some questions genuinely require
two stores simultaneously. No routing strategy can route a hybrid question to a single
store and get a complete answer.

---

## v4 — Multi-Store Agentic RAG

**Directory:** [`v4_multistore_rag/`](v4_multistore_rag/)

### What was designed and built
A verified relational store replaced LLM-extracted JSON for spec retrieval. Four PostgreSQL
tables hold the compressor's operational parameters, pressure settings, service plans, and
protection thresholds — seeded from authoritative CSV files. A Text-to-SQL retrieval node
generates and executes SQL against these tables. Hybrid retrieval combines SQL results with
graph context for compound questions.

### Architectural concepts
- Text-to-SQL retrieval — LLM generates SQL; database executes it; results are authoritative
- Schema prompting — explicit table definitions in the prompt constrain SQL generation
- Hybrid retrieval path — combines two stores in a single retrieval step
- Sentinel filtering — distinguishes empty results from genuine answers

### Technical decisions
- `SELECT *` always — avoids hallucinated column names
- `pg_conn.autocommit = True` — no transaction wrapping needed for read-only queries
- Seed CSVs committed — retrieval correctness is verifiable against source data
- `QUERY_TYPE_TO_STORE` mapping — explicit store assignment per query type

### Limitation that motivated the next version
Every answer attempt is evaluated only at the end — if the answer is wrong, the entire
retrieval was wasted. There is no mechanism to detect a poor answer and try again with
a different strategy.

---

## v5 — Self-Evaluating RAG

**Directory:** [`v5_self_evaluating_rag/`](v5_self_evaluating_rag/)

### What was designed and built
An LLM-as-judge node evaluates every generated answer and scores it on a 0–1 scale.
A conditional retry loop re-retrieves and re-generates if the score is insufficient.
Five exit conditions prevent infinite loops: good verdict, max attempts, score below
threshold, score declining across attempts, or judge parse error.

### Architectural concepts
- LLM-as-judge — a second LLM call evaluates the first; separation of generation and evaluation
- Score trend tracking — `score_trend` list detects improving vs declining quality across retries
- Adaptive stopping — five distinct exit conditions, each justified by a different failure mode
- Context accumulation — each retry appends to `accumulated_context`, not replace it

### Technical decisions
- Pydantic `AgentState` replaces TypedDict — field-level validation, cleaner partial returns
- `app.invoke()` returns dict — wrap with `AgentState(**result)` in tests
- `judge_parse_error` flag — distinguishes unparseable JSON from a genuine "insufficient" verdict
- Score trend of length ≥ 2 required before applying declining trend exit

### Limitation that motivated the next version
The retry strategy always queries the same store as the original attempt. When a store
is demonstrably insufficient — sparse graph edges, garbled OCR pages — retrying the
same store adds cost without improving answers. Cross-store fallback and smarter
re-retrieval strategy are the natural next step.

---

## v5.5 — Cross-Store Fallback and Smarter Graph Retrieval

**Directory:** [`v5_5_cross_store_rag/`](v5_5_cross_store_rag/)

### What was designed and built
Two targeted improvements to the retrieval and retry strategy. First, hybrid graph node
matching combines substring matching with semantic similarity using pre-computed node
embeddings — finding semantically related nodes that exact phrase matching misses.
Second, cross-store fallback routes retries to the most appropriate store based on the
judge's evaluation of what is missing, rather than always retrying the same store.

### Architectural concepts
- Pre-computed node embeddings — 1391 node labels embedded once at setup, loaded as
  NumPy matrix at import time; no embedding API calls per query
- Hybrid node matching — substring precision union semantic recall, capped at 5 seed nodes
- Vectorised cosine similarity — NumPy matrix operations over all nodes simultaneously
- Cross-store fallback — LLM reads judge evaluation and selects best store for retry
- Store capability prompting — explicit store descriptions guide LLM routing decision

### Technical decisions
- `embed_documents()` for batch embedding at setup; `embed_query()` for single query
- NumPy matrix `(1391, 1536)` loaded at import — no per-query conversion overhead
- `+ 1e-10` epsilon in cosine similarity denominator prevents division by zero
- Node existence guard `if node not in _G` defensive against version mismatch
- `fallback_store` field empty by default — distinguishes no-retry from retry clearly
- Semantic threshold 0.6 — balances precision and recall for short noisy node labels

### Limitation that motivated the next version
Every conversation starts from scratch. The agent has no memory of previous sessions —
a maintenance engineer must repeat queries across conversations, and the system
cannot learn from interaction history. Cross-session persistence is the missing capability.

---

## v6 — Long-term Memory and Cross-Session Persistence

**Directory:** [`v6_longterm_memory/`](v6_longterm_memory/)

### What was designed and built
A session-aware maintenance assistant with two complementary persistence mechanisms.
LangGraph `PostgresSaver` checkpointing saves and restores the full `AgentState` across
process boundaries, keyed by a `session_id`. An episodic memory store (`memory_episodes`
table, pgvector similarity search) records past Q&A cycles and injects semantically
similar past interactions as context at the start of every run.

### Architectural concepts
- LangGraph checkpointing — `PostgresSaver` serialises full `AgentState` to PostgreSQL
  across process boundaries; three internal tables (`checkpoints`, `checkpoint_blobs`,
  `checkpoint_writes`) provide atomic, recoverable state persistence
- Episodic memory — structured Q&A history queryable by semantic similarity via pgvector
- Bookend pattern — two new nodes wrap the unchanged v5.5 pipeline without modifying its internals
- Quality gate — only episodes scoring above `min_score_threshold` are persisted; low-quality
  answers discarded at write time rather than filtered at read time
- Embedding reuse — query embedding computed once in `memory_retrieval_node`, stored on
  `AgentState`, reused in `memory_storage_node` — eliminates redundant API call and
  guarantees embedding consistency within a single run

### Technical decisions
- `PostgresSaver` requires `psycopg3` (`psycopg[binary]`) — separate from `psycopg2`
  used by application retrieval nodes
- `session_id` on `AgentState` and `thread_id` in LangGraph config always set to same value
- `ivfflat` index omitted from `memory_episodes` — requires minimum row count to build lists;
  sequential scan correct at this scale
- `MemoryEntry.embedding` excluded from Pydantic model — embeddings passed separately
  to keep in-memory state lightweight
- `checkpointer.setup()` idempotent — safe to call on every module load

### Limitation that motivated the next version
v6 adds memory but has no visibility into what happened during a run. There is no trace
of which nodes fired, how long each took, what prompts were sent to the LLMs, or why
the judge scored an answer a particular way. Debugging requires adding print statements
and re-running. Observability is the missing capability.

---

## v7 — Observability, Tracing, and Gradio UI

**Directory:** [`v7_observability/`](v7_observability/)

### What was designed and built
Full-stack observability over the v6 agent, plus a production-ready Gradio UI. Every
LangGraph node is wrapped with an OpenTelemetry `@traced_node` decorator factory that
captures execution time and node-specific diagnostic attributes. Spans are sent to Jaeger
via OTLP. LangSmith captures all LLM calls automatically via environment variables.
A two-tab Gradio interface separates the end-user chatbot from the operator trace dashboard.

### Architectural concepts
- OTel decorator factory — `traced_node(span_name)` wraps nodes without touching their logic
- Child spans — helper functions open their own spans via `get_tracer()`; OTel context
  propagation nests them automatically under the parent node span
- LangSmith auto-instrumentation — zero instrumentation code in nodes; activated by env vars
- Separation of concerns — `observability/` module is completely independent of node logic
- Opt-in observability — both backends gracefully no-op when credentials are absent
- Honest failure — system correctly reports when source document does not contain an answer,
  rather than hallucinating plausible-sounding information

### Technical decisions
- `@functools.wraps(fn)` preserves `fn.__name__` — LangGraph uses function names during graph construction
- `BatchSpanProcessor` buffers spans for efficient network delivery to Jaeger
- `get_tracer()` imported lazily inside helpers — avoids circular import at module load time
- `sql[:500]` truncation on SQL span attribute — respects OTel attribute size limits
- `datetime.now(timezone.utc)` replaces deprecated `datetime.utcnow()` throughout
- `docker-compose.yml` manages all three containers (ChromaDB, PostgreSQL, Jaeger) together

### Limitation that motivated the next version
The evaluation framework is manual — a human must inspect traces to assess answer quality.
There is no automated evaluation suite that runs a fixed question set, scores answers against
ground truth, and tracks quality over time. v8 will introduce systematic evaluation.

---

## Shared Utilities

**Directory:** [`shared/`](shared/)

`pdf_loader.py` — PDF extraction cascade and semantic chunking. Used by v1, v2, and v2.5.
Contains:
- `load_pdf()` — per-page extraction with PyMuPDF → pdfplumber → OCR cascade
- `load_pdf_semantic()` — semantic chunking using cosine similarity between adjacent
  sentence embeddings

---

## Architectural Principles

These principles run through every version in the progression:

**Diagnosis before fix** — understand what's wrong and why before changing code.
A passing test with a wrong answer is more dangerous than a failing test with a clear
error message.

**Separation of concerns** — topology in `agent.py`, implementation in node files,
infrastructure in `clients.py`, verification in tests. Each file has one job.

**Module-level initialisation** — database connections, graph loading, LLM clients are
created once at import time and reused. Performance and predictability.

**Ingestion and retrieval are coupled** — what you can do at query time is determined
by what you stored at ingest time. Metadata filters, content type classification,
chunk boundaries — all decided during ingestion.

**Honest failure over confident hallucination** — "If the context does not contain
enough information, say so" is an architectural principle, not just a prompt instruction.
A system that admits uncertainty is safer in an industrial context than one that always
produces an answer.

**Progressive complexity with justified steps** — each version adds exactly one new
architectural concept. Complexity is never added for its own sake — only in response
to a demonstrated limitation of the previous solution.

**Observability as a first-class concern** — a system that cannot be observed cannot
be trusted in production. Tracing, latency visibility, and LLM call inspection are
architectural requirements, not afterthoughts.

---

## Infrastructure

All versions share the same infrastructure:

- **ChromaDB** — Docker container, `chromadb/chroma:0.5.23`, port 8000,
  collection `maintenance_manuals`
- **PostgreSQL** — Docker container, `postgres:15`, port 5432, database `compressor`
  (v4–v5.5); `ankane/pgvector` image required from v6 onwards (pgvector extension)
- **Jaeger** — Docker container, `jaegertracing/all-in-one`, port 16686 (UI), 4317 (OTLP) — v7 onwards
- **Embeddings** — `text-embedding-3-small` (OpenAI)
- **Generation** — `gpt-4o` (OpenAI)
- **Classification / extraction** — `gpt-4o-mini` (OpenAI)
- **Python** — 3.10.10, conda env `langchain-rag`
- **Key libraries** — LangChain 0.3.7, LangGraph, NetworkX 3.x, ChromaDB, psycopg2

From v7 onwards, all containers are managed together via `docker-compose.yml`.

---

## Source Document

Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03.
Sourced from ManualsLib for educational and research purposes:
https://www.manualslib.com/manual/1353128/Atlas-Copco-Ga5.html

---

## Running the Code

Each version directory contains the source code for that architectural stage.
To run any version, you will need:

- Python 3.10.10, conda environment with dependencies installed
- OpenAI API key set in a `.env` file (see `.env.example` in individual version directories)
- ChromaDB running as a Docker container on port 8000
- PostgreSQL running as a Docker container on port 5432 (v4 onwards)
- The Atlas Copco GA5 manual PDF (sourced from ManualsLib)

From v7 onwards, start all containers with `docker-compose up -d`.

Detailed setup instructions are in each version's `README.md`.

---

## Coming Next

**v8 — Evaluation**
Introduce a systematic evaluation framework. A fixed question set with ground-truth answers
runs against the agent automatically, scores are tracked across versions, and regressions
are detected before they reach production. The current system has no way to measure whether
architectural changes improve or degrade answer quality.
