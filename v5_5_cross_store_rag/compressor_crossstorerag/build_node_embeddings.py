"""
build_node_embeddings.py — Pre-compute embeddings for all graph nodes.

Run once before first use:
    python -m compressor_crossstorerag.build_node_embeddings

Why pre-compute:
    Semantic node matching requires comparing the query embedding against
    all node label embeddings. With 1391 nodes, computing embeddings at
    query time would add significant latency per query. Pre-computing once
    at setup time follows the same principle as module-level client
    initialisation — pay the cost once, reuse for every query.

Output:
    data/node_embeddings.json — dict mapping node label → embedding vector
    {
        "air filter": [0.123, -0.456, ...],   # 1536 dimensions
        "moisture prevention": [...],
        ...
    }

Cost: approximately $0.001 for 1391 nodes using text-embedding-3-small.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import networkx as nx
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

_GRAPH_PATH = Path(__file__).parent.parent / "data" / "graph.json"
_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "node_embeddings.json"


def build_node_embeddings() -> None:
    # Load graph
    print("Loading graph...")
    with open(_GRAPH_PATH) as f:
        G: nx.DiGraph = nx.node_link_graph(json.loads(f.read()), edges="edges")
    nodes = list(G.nodes)
    print(f"  {len(nodes)} nodes found.")

    # Extract node labels as strings
    node_labels = [str(n) for n in nodes]

    # Embed all node labels in a single API call
    print("Computing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(node_labels)
    print(f"  {len(vectors)} embeddings computed.")

    # Build mapping: node label → embedding vector
    node_embeddings = {
        label: vector
        for label, vector in zip(node_labels, vectors)
    }

    # Save to JSON
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(node_embeddings, f)
    print(f"  Saved to {_OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    build_node_embeddings()
