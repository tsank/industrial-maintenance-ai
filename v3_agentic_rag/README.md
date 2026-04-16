# compressor-agenticrag

> This repository is part of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.

v3 of the Industrial Maintenance AI architectural progression — Agentic RAG with conditional routing over the Atlas Copco GA5 compressor manual.

## What this project demonstrates

**Agentic routing with conditional edges** — a LangGraph `StateGraph` that classifies each incoming query and routes it to the most appropriate retrieval tool based on the nature of the question.

Three distinct retrieval strategies are integrated: semantic vector search for procedural and narrative queries, graph traversal for relationship and dependency queries, and structured keyword lookup for precise numeric specifications.

## Architecture
```
User query
    │
    ▼
classify_node        ← GPT-4o-mini classifies query type
    │
    ▼
route_query()        ← routing function reads query_type from state
    │
    ├── "spec"   → spec_retrieve_node    ← keyword match over specs.json
    ├── "graph"  → graph_retrieve_node   ← NetworkX graph traversal
    └── "vector" → vector_retrieve_node  ← ChromaDB semantic search
                        │
                        ▼
                   generate_node         ← GPT-4o generates answer from context
                        │
                        ▼
                      Answer
```

**Key LangGraph concepts:**
- `add_conditional_edges` — branching based on state content
- Routing functions — plain Python functions that read state and return a node name
- Multi-path graphs — fan-out from a decision point, fan-in to a common node

## The three retrieval paths

| Query type | Example query | Retriever | Data source |
|---|---|---|---|
| `spec` | "what is the shutdown temperature?" | Keyword token matching | `data/specs.json` |
| `graph` | "what components are involved in motor overload?" | NetworkX graph traversal | `data/graph.json` |
| `vector` | "how do I reset the service timer?" | ChromaDB similarity search | Semantic chunks in ChromaDB |

## Project structure
```
compressor-agenticrag/
├── compressor_agenticrag/
│   ├── __init__.py
│   ├── agent.py           # LangGraph StateGraph — core of this project
│   ├── pipeline.py        # Interactive entry point
│   ├── spec_extractor.py  # One-time extraction of specs from manual
│   └── spec_retriever.py  # Keyword-based spec lookup
├── data/
│   └── specs.json         # 34 structured specs extracted from the manual
├── tests/
│   ├── __init__.py
│   └── test_questions.py  # 5 test questions covering all three routing paths
├── .env.example
├── .gitignore
└── README.md
```

## Where this fits in the progression

| Version | Capability |
|---|---|
| v1 | Vector RAG — semantic search over manual |
| v2 | Graph RAG — knowledge graph + vector |
| v2.5 | Semantic chunking — better graph quality |
| **v3** | **Agentic routing — classify and route to right retriever** |

## Prerequisites

- Python 3.10+
- ChromaDB running as Docker container on port 8000 with collection `maintenance_manuals` populated
- OpenAI API key

## Setup
```bash
git clone https://github.com/tsank/compressor-agenticrag
cd compressor-agenticrag
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

Copy the knowledge graph from your graph store:
```bash
cp /path/to/graph.json data/graph.json
```

Start ChromaDB:
```bash
docker run -d --name chromadb -p 8000:8000 chromadb/chroma:0.5.23
```

## Usage

Interactive mode:
```bash
python -m compressor_agenticrag.pipeline
```

Run tests:
```bash
python -m tests.test_questions
```

## Test results
```
Results: 5/5 passed, 0/5 failed
```

| Test | Query | Route | Result |
|---|---|---|---|
| 1 | what is the compressor element outlet temperature shutdown level | spec | PASS |
| 2 | what is the nominal loading pressure for 13 bar | spec | PASS |
| 3 | what components are involved in a motor overload shutdown | graph | PASS |
| 4 | how do I modify the unloading pressure setting | vector | PASS |
| 5 | how does automatic restart after voltage failure work | vector | PASS |

## Data Source

Atlas Copco GA5 User Manual, Doc No. 2920 1461 03. Retrieved from [ManualsLib](https://www.manualslib.com) for educational purposes.

## Attribution

Built as v3 of the [Industrial Maintenance AI](https://github.com/tsank/industrial-maintenance-ai) architectural progression.
Developed with assistance from Claude (Anthropic).
