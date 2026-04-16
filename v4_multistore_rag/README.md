# compressor-multistorerag

**Problem 4 in a LangChain/LangGraph learning progression** — Multi-Store Agentic RAG for the Atlas Copco GA5 air compressor manual.

Demonstrates why four retrieval stores are needed, and why no single store can answer all query types correctly.

---

## Architecture

```
classify_node
     │
     ▼
route_query() ──── spec      ──► db_retrieve_node     ──┐
                ├─ procedure ──► vector_retrieve_node  ──┤
                ├─ relation  ──► graph_retrieve_node   ──┤
                └─ hybrid    ──► hybrid_retrieve_node  ──┘
                                                         │
                                                    synthesise_node
                                                         │
                                                    generate_node
```

### The four stores and why each is necessary

| Store | Query type | Why it can't be replaced |
|---|---|---|
| **PostgreSQL** | `spec` | Exact structured values (thresholds, intervals, pressure setpoints) — semantic similarity cannot reliably return `120 °C` as a fact |
| **ChromaDB** | `procedure` | Multi-sentence procedural steps and safety narrative — no SQL schema can capture free-text instructions |
| **NetworkX** | `relation` | Component relationships and system dependencies — edges between entities can't be stored in SQL rows or retrieved by cosine similarity |
| **PostgreSQL + NetworkX** | `hybrid` | Service plans need exact intervals (SQL) *and* component involvement (graph); neither alone is sufficient |

### AgentState

```python
class AgentState(TypedDict):
    query: str
    query_type: str       # spec | procedure | relation | hybrid
    sql_context: str      # populated by db_retrieve or hybrid_retrieve
    vector_context: str   # populated by vector_retrieve or hybrid_retrieve
    graph_context: str    # populated by graph_retrieve or hybrid_retrieve
    context: str          # merged by synthesise_node
    answer: str
```

---

## Setup

### 1. PostgreSQL container

```bash
docker run -d --name compressor-pg \
  -e POSTGRES_DB=compressor \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:15
```

### 2. ChromaDB container

Already running from Problems 1–3:
```bash
# collection maintenance_manuals must already exist
```

### 3. Dependencies

```bash
python -m pip install langchain==0.3.7 langgraph langchain-openai \
    chromadb psycopg2-binary python-dotenv networkx
```

### 4. Environment

```bash
cp .env.example .env
# fill in OPENAI_API_KEY
```

### 5. Copy graph from Problem 2

```bash
cp ../compressor-graphrag/data/graph.json data/graph.json
```

### 6. Create tables and load seed data

```bash
python -m compressor_multistorerag.db_setup
```

---

## Run tests

```bash
python -m pytest tests/ -v -s
```

---

## Data source

Manual: Atlas Copco GA5 Instruction Book, Document No. 2920 1461 03, sourced from ManualsLib.  
Seed CSV values are manually verified against the manual. No LLM extraction was used for structured data.