# compressor_agenticrag/agent.py
from pathlib import Path
from typing import TypedDict
import chromadb

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
import networkx as nx
import json

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from compressor_agenticrag.spec_retriever import retrieve_specs, format_spec_context

GRAPH_PATH = Path(__file__).parent.parent / "data" / "graph.json"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "maintenance_manuals"


# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    query_type: str
    context: str
    answer: str


# ── Clients (initialised once at module load) ──────────────────────────────────

_llm_classify = ChatOpenAI(model="gpt-4o-mini", temperature=0)
_llm_generate = ChatOpenAI(model="gpt-4o", temperature=0)

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
_chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
_vectorstore = Chroma(
    client=_chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
)

_graph_data = nx.node_link_graph(
    json.loads(GRAPH_PATH.read_text()),
    edges="edges"
)


# ── Classification prompt ──────────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are classifying a maintenance query about an Atlas Copco compressor.

Classify the query into exactly one of these three types:

- "spec": The query asks for a specific numeric value, threshold, rating, or setting.
  Examples: "what is the shutdown temperature", "maximum unloading pressure",
  "service interval", "motor voltage"

- "graph": The query asks about relationships, dependencies, or what connects to what.
  Examples: "what triggers a shutdown", "what components are involved in motor overload",
  "what conditions cause the alarm LED to light up"

- "vector": The query asks how to do something, explains a procedure, or asks for
  contextual/narrative information.
  Examples: "how do I reset the service timer", "what happens during automatic restart",
  "how does the regulator control the compressor"

Reply with exactly one word: spec, graph, or vector.

Query: {query}"""


# ── Nodes ──────────────────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    prompt = CLASSIFY_PROMPT.format(query=state["query"])
    response = _llm_classify.invoke(prompt)
    query_type = response.content.strip().lower()

    # Defensive fallback if model returns something unexpected
    if query_type not in ("spec", "graph", "vector"):
        query_type = "vector"

    print(f"  [classify] query_type = {query_type}")
    return {**state, "query_type": query_type}


def vector_retrieve_node(state: AgentState) -> AgentState:
    print("  [retrieve] using vector search")
    docs = _vectorstore.similarity_search(state["query"], k=4)
    context = "\n\n".join(d.page_content for d in docs)
    return {**state, "context": context}


def graph_retrieve_node(state: AgentState) -> AgentState:
    print("  [retrieve] using graph traversal")
    query_terms = set(state["query"].lower().split())

    scored_nodes = []
    for node in _graph_data.nodes():
        score = sum(1 for term in query_terms if term in node.lower())
        if score > 0:
            scored_nodes.append((node, score))

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    seed_nodes = [node for node, _ in scored_nodes[:5]]

    triples = []
    seen = set()
    for node in seed_nodes:
        for _, target, data in _graph_data.out_edges(node, data=True):
            key = (node, data.get("relation", "related_to"), target)
            if key not in seen:
                seen.add(key)
                triples.append(f"{node} {data.get('relation', 'related_to')} {target}")
        for source, _, data in _graph_data.in_edges(node, data=True):
            key = (source, data.get("relation", "related_to"), node)
            if key not in seen:
                seen.add(key)
                triples.append(f"{source} {data.get('relation', 'related_to')} {node}")

    context = "Graph relationships:\n" + "\n".join(triples) if triples else "No graph results found."
    return {**state, "context": context}


def spec_retrieve_node(state: AgentState) -> AgentState:
    print("  [retrieve] using spec lookup")
    matches = retrieve_specs(state["query"], top_k=3)
    context = format_spec_context(matches)
    return {**state, "context": context}


def generate_node(state: AgentState) -> AgentState:
    prompt = f"""You are a helpful assistant for Atlas Copco compressor maintenance.
Use the following context to answer the question. Be concise and precise.
If the context does not contain enough information, say so clearly.

Context:
{state["context"]}

Question: {state["query"]}

Answer:"""
    response = _llm_generate.invoke(prompt)
    return {**state, "answer": response.content.strip()}


# ── Routing function ───────────────────────────────────────────────────────────

def route_query(state: AgentState) -> str:
    return state["query_type"]


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("vector_retrieve", vector_retrieve_node)
    graph.add_node("graph_retrieve", graph_retrieve_node)
    graph.add_node("spec_retrieve", spec_retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_query,
        {
            "vector": "vector_retrieve",
            "graph": "graph_retrieve",
            "spec": "spec_retrieve",
        }
    )

    graph.add_edge("vector_retrieve", "generate")
    graph.add_edge("graph_retrieve", "generate")
    graph.add_edge("spec_retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── Public entry point ─────────────────────────────────────────────────────────

def run(query: str) -> dict:
    agent = build_graph()
    result = agent.invoke({"query": query, "query_type": "", "context": "", "answer": ""})
    return result