"""
clients.py — Module-level client initialisation shared across all nodes.

All connections and models are initialised once at import time.
Every node file imports what it needs from here — no node initialises
its own clients.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import networkx as nx
import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# LLMs and embeddings
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)
classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_CHROMA_HOST = os.environ["CHROMA_HOST"]
_CHROMA_PORT = int(os.environ["CHROMA_PORT"])
_CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "maintenance_manuals")

chroma_client = chromadb.HttpClient(host=_CHROMA_HOST, port=_CHROMA_PORT)
collection = chroma_client.get_collection(_CHROMA_COLLECTION)

# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

pg_conn = psycopg2.connect(
    host=os.environ["PG_HOST"],
    port=int(os.environ.get("PG_PORT", 5432)),
    dbname=os.environ["PG_DB"],
    user=os.environ["PG_USER"],
    password=os.environ["PG_PASSWORD"],
)
pg_conn.autocommit = True

# ---------------------------------------------------------------------------
# NetworkX graph
# ---------------------------------------------------------------------------

_GRAPH_PATH = Path(__file__).parent.parent.parent / "data" / "graph.json"
with open(_GRAPH_PATH) as f:
    G: nx.DiGraph = nx.node_link_graph(json.loads(f.read()), edges="edges")

# ---------------------------------------------------------------------------
# Pre-computed node embeddings
# ---------------------------------------------------------------------------

_NODE_EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "node_embeddings.json"
with open(_NODE_EMBEDDINGS_PATH) as f:
    _node_embeddings: dict[str, list[float]] = json.load(f)

node_labels: list[str] = list(_node_embeddings.keys())
node_matrix: np.ndarray = np.array(list(_node_embeddings.values()))

# ---------------------------------------------------------------------------
# Observability setup
# ---------------------------------------------------------------------------

from compressor_observability.observability import setup_otel, setup_langsmith

setup_otel(service_name="industrial-maintenance-ai-v7")
setup_langsmith()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEMANTIC_THRESHOLD = 0.6

EMPTY_SIGNALS = {
    "No matching entities found in knowledge graph.",
    "Entities found but no relationships retrieved.",
    "No relevant passages found.",
    "No matching records found.",
    "SQL error",
}

QUERY_TYPE_TO_STORE = {
    "spec": "postgresql",
    "procedure": "chromadb",
    "relation": "networkx",
    "hybrid": "postgresql",
}

PG_SCHEMA = """
Tables in the compressor database:

operational_parameters(parameter TEXT PRIMARY KEY, value TEXT, unit TEXT)
  -- single-value parameters, e.g. maximum_working_pressure, oil_capacity

pressure_settings(ga_model TEXT, icd_model TEXT, dewpoint_variant TEXT,
                  frequency_hz INT, unload_pressure_bar FLOAT,
                  load_pressure_bar FLOAT)
  -- per-model pressure setpoints; ga_model is always 'GA5'

service_plans(plan CHAR(1), interval_hours INT, interval_months INT,
              description TEXT)
  -- service plans A/B/C/D with calendar and hour-based intervals

protection_thresholds(parameter TEXT, level TEXT, value FLOAT, unit TEXT)
  -- shutdown and warning thresholds, e.g. high_temperature shutdown at 120 degrees C
"""
