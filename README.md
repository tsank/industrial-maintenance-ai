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

**Repo:** [`compressor-multistorerag`](https://github.com/tsank/compressor-multistorerag)
**Directory:** [`v4_multistore_rag/`](v4_multistore_rag/)

### What was designed and built
A four-store retrieval system: PostgreSQL for structured tabular data (Text-to-SQL),
ChromaDB for narrative and procedural text (vector search), NetworkX for component
relationships (graph traversal), and hybrid retrieval paths that call two stores for
compound questions. A synthesis node merges multiple context fields before generation.

### Architectural concepts
- Text-to-SQL — schema-aware prompting generates SQL from natural language
- Hybrid retrieval — some questions require two stores; no single store can answer them
- `_extract_sql_question()` — strips non-SQL parts before SQL generation to prevent
  the LLM from generating incorrect joins across unrelated tables
- Synthesis node — merges populated context fields with labelled section headers
- Sentinel filtering — empty retrieval signals excluded before generation
- Verified seed data — safety-critical structured values loaded from human-verified CSVs

### Technical decisions
- Metadata filtering in ChromaDB only works if metadata was written at ingest time —
  retrieval capabilities are constrained by ingestion decisions
- Graph node matching direction matters — flipped node-in-query matching handles
  multi-word node labels correctly; token-in-node produces false positives
- Successor and predecessor directions must be preserved separately in graph output
- `IMPORTANT: Always use SELECT *` in the SQL prompt prevents the LLM from
  discarding column names that give values their meaning
- Debugging RAG systems requires tracing data through every stage before identifying
  root cause — passing tests with loose assertions are not proof of correctness

### The four-store argument
Four stores exist because four types of knowledge require four different access patterns:

**PostgreSQL** — exact structured facts (thresholds, intervals, pressure settings).
Semantic similarity returns approximately related content. SQL returns the exact row.
For a shutdown temperature threshold, approximate is not acceptable.

**ChromaDB** — narrative and procedural content. Step-by-step instructions exist as
prose across multiple sentences. There is no schema that captures arbitrary step
sequences. Vector search finds relevant passages regardless of exact wording.

**NetworkX** — component relationships. Edges between entities are first-class data
structures. SQL has no edges. Vector search returns prose about components, not
structured relationship data.

**Hybrid paths** — compound questions that span multiple knowledge types. A question
asking for both a service interval and the components involved requires PostgreSQL for
the interval and NetworkX for the components. Neither store alone can answer it.

---

## v5 — Self-Evaluating RAG

**Directory:** [`v5_self_evaluating_rag/`](v5_self_evaluating_rag/)

### What was designed and built
A self-evaluating agent that assesses its own answer quality after each generation
attempt and decides whether to retry, exit early, or accept the answer. A judge node
evaluates answer completeness after every generation. Context accumulates across retries.
The graph contains a cycle for the first time in the progression.

### Architectural concepts
- LangGraph cycles — graph contains a loop; `synthesise → generate → judge → re_retrieve → synthesise`
- LLM-as-judge — second LLM call evaluates answer quality after each generation
- Score trend tracking — scores recorded across attempts detect improving vs declining retrieval
- Adaptive stopping — five distinct exit conditions based on verdict, score, parse errors, and trend
- Context accumulation — `accumulated_context` grows with each retry; later attempts have richer context
- Targeted re-retrieval — judge's `evaluation` field drives focused follow-up queries
- Pydantic AgentState — runtime type enforcement replacing TypedDict; critical with cycles

### Technical decisions
- `judge_parse_error` flag gates all score-based exit logic — malformed JSON retries unconditionally
- `min_score_threshold = 0.3` prevents pointless retries when content is absent from the knowledge base
- Single-step score trend check is appropriate for `max_attempts = 3`
- `previous_answer` saved by `generate_node` before overwriting — recoverable when trend declines
- `app.invoke()` returns a dict — wrap with `AgentState(**result)` for Pydantic attribute access

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

### Limitation acknowledged but not addressed in subsequent versions
Cross-store fallback selection by LLM is probabilistic. Improving it would require
historical routing performance data or domain-specific fine-tuning — neither of which
fits the single-concept progression structure. This remains an open improvement area.

### Limitation that motivated the next version
Every conversation starts from scratch. The agent has no memory of previous sessions —
a maintenance engineer must repeat queries across conversations, and the system
cannot learn from interaction history. Cross-session persistence is the missing capability.

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

**Separation of concerns** — topology in `agent.py`, implementation in `pipeline.py`,
infrastructure in `db_setup.py`, verification in `test_questions.py`. Each file has
one job.

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

---

## Infrastructure

All versions share the same infrastructure:

- **ChromaDB** — Docker container, `chromadb/chroma:0.5.23`, port 8000,
  collection `maintenance_manuals`
- **PostgreSQL** — Docker container, `postgres:15`, port 5432, database `compressor`
  (v4 onwards)
- **Embeddings** — `text-embedding-3-small` (OpenAI)
- **Generation** — `gpt-4o` (OpenAI)
- **Classification / extraction** — `gpt-4o-mini` (OpenAI)
- **Python** — 3.10.10, conda env `langchain-rag`
- **Key libraries** — LangChain 0.3.7, LangGraph, NetworkX 3.x, ChromaDB, psycopg2

---

## Source Document

Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03.
Sourced from ManualsLib for educational and research purposes:
https://www.manualslib.com/manual/1234567/Atlas-Copco-Ga5.html

---

## Running the Code

Each version directory contains the source code for that architectural stage.
To run any version, you will need:

- Python 3.10.10, conda environment with dependencies installed
- OpenAI API key set in a `.env` file (see `.env.example` in individual version directories)
- ChromaDB running as a Docker container on port 8000
- PostgreSQL running as a Docker container on port 5432 (v4 onwards)
- The Atlas Copco GA5 manual PDF (sourced from ManualsLib)

Detailed setup instructions are in each version's `README.md`.

---

## Coming Next

**v6 — Long-term Memory and Cross-Session Persistence**
Introduce LangGraph checkpointing for cross-session state persistence.
The agent remembers previous queries and answers across conversations,
building a session-aware maintenance assistant.
