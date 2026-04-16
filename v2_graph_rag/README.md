# Compressor GraphRAG

A LangChain/LangGraph learning project focused on industrial AI applications — Problems 2 and 2.5 in a four-problem progression.

## Overview

A GraphRAG system over the Atlas Copco GA5 compressor manual. Extracts entities and relationships into a knowledge graph, combines graph traversal with vector retrieval, and compares answer quality against [maintenance-rag](https://github.com/tsank/maintenance-rag) (basic RAG).

Problem 2.5 introduced semantic chunking — replacing fixed-size chunking with embedding-based sentence similarity splitting. Both ChromaDB and the graph builder now consume the same 136 semantic chunks, making the architecture fully unified.

## Architecture
```
PDF (517ba4.pdf)
  └── load_pdf_robust()         # per-page extraction cascade (PyMuPDF → pdfplumber → OCR)
        └── load_pdf_semantic() # semantic chunking via cosine similarity (136 chunks)
              ├── ChromaDB      # vector store (text-embedding-3-small)
              └── graph_builder.py  # spaCy NER + GPT-4o-mini triple extraction → NetworkX graph

Query
  ├── ChromaDB (semantic vector search)
  ├── NetworkX (knowledge graph traversal)
  └── LangGraph fusion → GPT-4o → Answer
```

## Project Structure
```
compressor-graphrag/
├── compressor_graphrag/
│   ├── graph_builder.py      # semantic chunk extraction + knowledge graph construction
│   ├── graph_retriever.py    # keyword-scored node matching + 1-hop expansion
│   ├── hybrid_retriever.py   # merges ChromaDB vector + graph context
│   └── pipeline.py           # LangGraph 2-node pipeline (retrieve → generate)
├── data/
│   └── graph.json            # serialized knowledge graph (generated, not committed)
├── tests/
│   └── test_questions.py     # evaluation harness for 5 test questions
├── ingest_semantic.py        # semantic chunking ingestion pipeline
├── main.py
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10+
- conda environment: `langchain-rag`
- Docker Desktop running (for ChromaDB)
- `maintenance-rag` repo cloned as a sibling directory (provides `load_pdf_robust()` and `load_pdf_semantic()`)
- OpenAI API key

### Installation
```bash
conda activate langchain-rag
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Environment

Create a `.env` file at the repo root:
```
OPENAI_API_KEY=your_key_here
```

### ChromaDB

ChromaDB runs as a Docker container inherited from maintenance-rag. Start it from the `maintenance-rag` directory:
```bash
cd ../maintenance-rag
docker-compose up -d chromadb
```

## Usage

### Step 1 — Ingest semantic chunks into ChromaDB (run once)
```bash
python ingest_semantic.py
```

Loads the GA5 manual, applies semantic chunking (136 chunks), flushes the existing ChromaDB collection, and re-ingests. Costs approximately $0.001 in OpenAI API usage.

### Step 2 — Build the knowledge graph (run once)
```bash
python compressor_graphrag/graph_builder.py
```

Processes the 136 semantic chunks, extracts entities via spaCy and triples via GPT-4o-mini, and saves the graph to `data/graph.json`. Costs approximately $0.36 in OpenAI API usage.

### Step 3 — Run evaluation
```bash
python tests/test_questions.py
```

Runs 5 test questions through the full hybrid pipeline and prints answers with graph context used.

### Step 4 — Single query
```bash
python main.py
```

## Knowledge Graph Stats

| Metric | Problem 2 (fixed/page-level) | Problem 2.5 (semantic chunks) |
|--------|-------------------------------|-------------------------------|
| Input to graph builder | 49 pages | 136 semantic chunks |
| Total nodes | 1004 | 1391 |
| Total edges | 387 | 686 |
| ChromaDB chunks | 802 fixed | 136 semantic |
| Extraction method | spaCy NER + GPT-4o-mini | spaCy NER + GPT-4o-mini |

## How It Works

**Semantic chunking** — `load_pdf_semantic()` (in `maintenance-rag/src/pdf_loader.py`) embeds every sentence using `text-embedding-3-small`, computes cosine similarity between adjacent sentences, and splits where similarity drops below the 25th percentile. This produces topically coherent chunks without requiring `langchain-experimental`.

**Graph construction** — `graph_builder.py` runs once over the 136 semantic chunks. spaCy extracts named entities as nodes. GPT-4o-mini extracts `(subject, relation, object)` triples as edges. The result is a NetworkX directed graph serialized to JSON.

**Hybrid retrieval** — For each query, `graph_retriever.py` scores every graph node by keyword match, selects the top-5 seed nodes, and expands to their 1-hop neighborhood. Simultaneously, ChromaDB returns the top-4 semantically similar chunks. Both contexts are merged.

**Generation** — A LangGraph 2-node pipeline passes the merged context to GPT-4o for answer generation.

## Comparison Across Problems

| Aspect | Problem 1 — Basic RAG | Problem 2 — GraphRAG | Problem 2.5 — Semantic GraphRAG |
|--------|----------------------|----------------------|----------------------------------|
| Chunking | Fixed (802 chunks) | Page-level (49 pages) | Semantic (136 chunks) |
| Vector store | 802 fixed chunks | 802 fixed chunks | 136 semantic chunks |
| Graph input | — | 49 pages | 136 semantic chunks |
| Graph nodes | — | 1004 | 1391 |
| Graph edges | — | 387 | 686 |
| Retrieval | Vector only | Vector + graph | Vector + graph (unified) |
| Architecture | Simple | Misaligned | Fully unified |

## Key Learnings

- **Semantic chunking** groups sentences by meaning rather than character count — better retrieval context at lower chunk count
- **Unified architecture** matters: both ChromaDB and the graph builder should consume the same chunks for consistency
- **Graph extraction quality** improves with focused chunks — 136 semantic chunks produced 39% more nodes and 77% more edges than 49 pages
- **`langchain-experimental` version conflicts** with `langchain 0.3.7` — semantic chunking implemented directly using cosine similarity instead
- **GraphRAG strength** is most visible on relational queries (e.g. automatic restart function) where graph relationships directly support the answer

## Dependencies

- LangChain 0.3.7
- LangGraph
- NetworkX
- spaCy (en_core_web_sm)
- ChromaDB 0.5.23
- OpenAI (gpt-4o, gpt-4o-mini, text-embedding-3-small)
- numpy (cosine similarity for semantic chunking)

## Related

- [maintenance-rag — Basic RAG (Problem 1)](https://github.com/tsank/maintenance-rag)

## Data Source

The sample manual used for development and testing is the Atlas Copco GA5 User Manual (Document No. 2920 1461 03), sourced from ManualsLib:
https://www.manualslib.com/manual/1234567/Atlas-Copco-Ga5.html

## Attribution

Built as part of a self-directed LangChain/LangGraph learning progression focused on industrial AI applications.
Developed with assistance from Claude (Anthropic).
