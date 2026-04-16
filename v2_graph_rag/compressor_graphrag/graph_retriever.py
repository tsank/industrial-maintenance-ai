import sys
from pathlib import Path
import networkx as nx

from graph_builder import load_graph


def get_relevant_subgraph(query: str, G: nx.DiGraph, top_n: int = 5) -> list[dict]:
    query_terms = set(query.lower().split())

    scored_nodes = []
    for node in G.nodes():
        node_lower = node.lower()
        score = sum(1 for term in query_terms if term in node_lower)
        if score > 0:
            scored_nodes.append((node, score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    seed_nodes = [node for node, _ in scored_nodes[:top_n]]

    triples = []
    for node in seed_nodes:
        for _, target, data in G.out_edges(node, data=True):
            triples.append({
                "subject": node,
                "relation": data.get("relation", "related_to"),
                "object": target
            })
        for source, _, data in G.in_edges(node, data=True):
            triples.append({
                "subject": source,
                "relation": data.get("relation", "related_to"),
                "object": node
            })

    seen = set()
    unique_triples = []
    for t in triples:
        key = (t["subject"], t["relation"], t["object"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    return unique_triples


def triples_to_context(triples: list[dict]) -> str:
    if not triples:
        return "No graph context found."
    lines = [f"{t['subject']} {t['relation']} {t['object']}" for t in triples]
    return "Graph relationships:\n" + "\n".join(lines)


if __name__ == '__main__':
    G = load_graph()
    query = "What is the role of the Elektronikon regulator?"
    triples = get_relevant_subgraph(query, G)
    print(triples_to_context(triples))