# v5.5 — Cross-Store Fallback and Smarter Graph Retrieval

> This repository is part of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.

v5.5 of the Industrial Maintenance AI architectural progression — Cross-Store Fallback and Semantic Graph Node Matching over the Atlas Copco GA5 compressor manual.

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
| **v5.5** | **Cross-Store Fallback — smarter graph matching + intelligent retry routing** |

---

## Architectural limitation addressed from v5

v5 had two specific weaknesses in its retrieval strategy:

**1. Same-store retry**
`re_retrieve_node` always retried from the same store as the original query. When a store was demonstrably insufficient — sparse graph edges, garbled OCR pages — retrying added API cost without improving answers.

**2. Substring-only graph node matching**
`_graph_search()` required node labels to appear as exact substrings in the query. Multi-word related concepts — "air inlet" when querying about "inlet air system" — were missed entirely. This produced sparse graph context even when semantically relevant nodes existed.

---

## What was designed and built

Two targeted improvements to the retrieval and retry strategy:

**Hybrid graph node matching** — substring matching combined with semantic similarity matching using pre-computed node embeddings. The union of both approaches finds more relevant nodes while preserving the precision of exact phrase matching.

**Cross-store fallback** — when `re_retrieve_node` triggers a retry, an LLM reads the judge's evaluation of what is missing and selects the most appropriate store for the follow-up query — not necessarily the same store as the original.

---

## Architecture

Graph topology is identical to v5. All changes are inside `pipeline.py`:

```
classify_node
      │
      ▼
route_query() ── conditional
      │
      ├── spec      → db_retrieve_node
      ├── procedure → vector_retrieve_node
      ├── relation  → graph_retrieve_node   ← uses hybrid matching
      └── hybrid    → hybrid_retrieve_node  ← uses hybrid matching
                            │
                       synthesise_node  ◄───────────────────--───┐
                            │                                    │
                       generate_node                             │
                            │                                    │
                        judge_node                               │
                            │                                    │
                   should_retry() ── conditional                 │
                            │                                    │
                ┌── retry ──────► re_retrieve_node ──────────-───┘
                │                  │
                │                  └── _select_fallback_store()
                │                      LLM selects: postgresql | chromadb | networkx
                │
                └── end ──► END
```

---

## New in v5.5

### 1. Pre-computed node embeddings

A one-time setup script embeds all 1391 graph node labels using `text-embedding-3-small` and saves them to `data/node_embeddings.json`:

```bash
python -m compressor_crossstorerag.build_node_embeddings
```

At module import time, `pipeline.py` loads these embeddings as a NumPy matrix `(1391, 1536)` — ready for vectorised cosine similarity computation without any API calls per query.

**Why pre-compute rather than compute at query time:**
With 1391 nodes, computing embeddings at query time would add significant latency. Pre-computing follows the same principle as module-level client initialisation throughout the progression — pay the cost once at startup, reuse for every query.

---

### 2. Hybrid graph node matching

`_graph_search()` now uses two matching strategies in union:

```
Step 1 — Substring matching (existing)
    matched_substring = {n for n in G.nodes
                         if len(str(n)) > 4 and str(n).lower() in query_lower}

Step 2 — Semantic matching (new)
    query_vec = embed(query)                          # one API call
    similarities = dot(node_matrix, query_vec) /      # vectorised, no loop
                   (norm(node_matrix) * norm(query_vec))
    matched_semantic = {nodes above 0.6 threshold, top-5 by score}

Step 3 — Union, capped at 5 seed nodes
    matched = (matched_substring ∪ matched_semantic)[:5]

Step 4 — Direction-preserving traversal (unchanged from v5)
    successors: node → neighbour
    predecessors: neighbour → node
```

**Why union rather than intersection:**
Intersection would only return nodes matched by both methods — very restrictive. Union gives the precision of substring matching for exact phrases and the recall of semantic matching for related concepts.

**Why 0.6 as the threshold:**
Below 0.5 — too loose, matches unrelated concepts. Above 0.8 — too strict, barely better than substring matching. 0.6 balances precision and recall for short, noisy node labels.

**Test result:**
For query "What is connected to the inlet air system?":
- Substring matches: 2 nodes (`system`, `inlet air`)
- Semantic matches: 4 nodes (`recirculate to the inlet`, `inlet air`, `air inlet`, `air inlet and outlet systems`)
- Union: 5 nodes — significantly richer neighbourhood context

---

### 3. Cross-store fallback

When `re_retrieve_node` triggers, `_select_fallback_store()` asks the LLM:

```
Original query type: relation
Original store used: networkx
Judge evaluation — what is missing: does not identify components associated with air filter

Which store is most likely to contain the missing information?
- postgresql: exact parameter values, thresholds, service intervals
- chromadb:   narrative procedures, safety instructions, step-by-step text
- networkx:   component relationships, system dependencies

Reply with exactly one word: postgresql, chromadb, or networkx.
```

The LLM's response routes the retry to the most appropriate store — not the original store.

**Safety fallback:** if the LLM returns an unexpected value, the function falls back to the original store. The retry always happens.

---

## AgentState — one new field

```python
class AgentState(BaseModel):
    # ... all v5 fields unchanged ...
    fallback_store: str = ""    # store selected by LLM for retry
                                # "postgresql" | "chromadb" | "networkx" | ""
```

`fallback_store` is populated only when a retry occurs. Empty string when no retry happened — either good verdict on first attempt or early exit.

---

## Exit conditions — unchanged from v5

```
verdict == "good"
    → EXIT with current answer

verdict == "insufficient" AND attempt >= max_attempts
    → EXIT with current answer

verdict == "insufficient" AND valid JSON AND score < min_score_threshold
    → EXIT with current answer (content not in knowledge base)

verdict == "insufficient" AND parse error
    → RETRY unconditionally

verdict == "insufficient" AND valid JSON AND score declining
    → EXIT with previous_answer

verdict == "insufficient" AND valid JSON AND score stable/improving
    → RETRY

verdict == "insufficient" AND no trend yet
    → RETRY
```

---

## Known limitations

**Cross-store fallback selection is probabilistic**
The LLM selects the fallback store based on the judge's natural language evaluation. This selection is not always correct — "does not identify components associated with the air filter" was interpreted as needing structured data (PostgreSQL) rather than narrative text (ChromaDB) or graph traversal (NetworkX). Improving fallback store selection would require richer store capability metadata, historical performance tracking, or fine-tuning on domain-specific routing examples.

**Semantic threshold is fixed**
The 0.6 cosine similarity threshold for node matching is a module-level constant. In production, this would be a configurable parameter tuned per domain and graph quality.

**Graph quality ceiling**
Semantic matching finds more nodes but is bounded by the quality of the graph itself. Nodes that were never extracted during triple extraction in v2 cannot be found regardless of matching strategy.

---

## Setup sequence

```bash
# 1. Copy .env from v5 or create from .env.example
cp .env.example .env
# Add OPENAI_API_KEY

# 2. Ensure Docker containers are running
docker ps   # should show chromadb and compressor-pg

# 3. Create tables and load seed data
python -m compressor_crossstorerag.db_setup

# 4. Pre-compute node embeddings (one-time)
python -m compressor_crossstorerag.build_node_embeddings

# 5. Run tests
python -m pytest tests/ -v -s
```

---

## Project structure

```
v5_5_cross_store_rag/
├── compressor_crossstorerag/
│   ├── __init__.py
│   ├── agent.py                  # StateGraph — identical to v5 + fallback_store field
│   ├── build_node_embeddings.py  # One-time setup: pre-compute node embeddings
│   ├── db_retriever.py           # Direct PostgreSQL access for tests
│   ├── db_setup.py               # Create tables and load seed data
│   └── pipeline.py               # Hybrid _graph_search(), _select_fallback_store(),
│                                 # updated re_retrieve_node()
├── data/
│   ├── node_embeddings.json      # Generated by build_node_embeddings.py (gitignored)
│   └── seed/
│       ├── operational_parameters.csv
│       ├── pressure_settings.csv
│       ├── protection_thresholds.csv
│       └── service_plans.csv
├── tests/
│   ├── __init__.py
│   └── test_questions.py         # 7 tests — 4 from v5 + 3 new v5.5 tests
├── .env.example
├── .gitignore
└── README.md
```

---

## Test results

```
7 passed
```

| Test | Query | Verdict | Score | Attempts | Fallback store |
|---|---|---|---|---|---|
| test_seed_data_loaded | Direct DB verification | — | — | — | — |
| test_spec_good | Shutdown temperature threshold | good | 1.0 | 1 | — |
| test_procedure_good | Outlet temperature warning steps | good | 1.0 | 1 | — |
| test_hybrid_good | Service intervals + components | good | 1.0 | 1 | — |
| test_semantic_node_matching | Inlet air system (direct) | — | — | — | — |
| test_cross_store_fallback | Air filter components | insufficient | 0.2 | 2 | postgresql |
| test_fallback_store_field | Max pressure + content gap | good/insufficient | 1.0/0.2 | 1/1 | "" |

---

## Architectural concepts

- Pre-computed embeddings — pay embedding cost once at setup, reuse at query time
- Hybrid node matching — substring precision union semantic recall
- Vectorised cosine similarity — NumPy matrix operations over 1391 nodes simultaneously
- Cross-store fallback — LLM-guided store selection for targeted retry
- Store capability prompting — explicit store descriptions guide LLM routing decision

## Technical decisions

- `embed_documents()` for batch embedding, `embed_query()` for single query embedding
- NumPy matrix `(1391, 1536)` loaded at import time — no per-query conversion overhead
- `+ 1e-10` epsilon in cosine similarity denominator — prevents division by zero
- Node existence guard `if node not in _G` — defensive against embedding/graph version mismatch
- `_QUERY_TYPE_TO_STORE` mapping used both in prompt and as safety fallback
- `fallback_store` empty string default — distinguishes no-retry from retry clearly

---

## Data Source

Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03.
Sourced from ManualsLib for educational and research purposes.

---

## Attribution

Built as v5.5 of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.
Developed with assistance from Claude (Anthropic).
