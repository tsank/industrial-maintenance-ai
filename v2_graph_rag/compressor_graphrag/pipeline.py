from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict
import networkx as nx
from langchain_chroma import Chroma

load_dotenv(Path(__file__).parent.parent / '.env', override=True)

from graph_builder import load_graph
from hybrid_retriever import load_vector_store, hybrid_retrieve

llm = ChatOpenAI(model='gpt-4o', temperature=0)


# ── 1. State definition ──────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    vector_context: str
    graph_context: str
    merged_context: str
    answer: str


# ── 2. Pipeline factory ───────────────────────────────────────────────────────

def build_pipeline(G: nx.DiGraph, vector_store: Chroma):
    """Build and return a compiled LangGraph pipeline."""

    def retrieve_node(state: RAGState) -> RAGState:
        result = hybrid_retrieve(state['query'], G, vector_store)
        return {
            **state,
            'vector_context': result['vector_context'],
            'graph_context': result['graph_context'],
            'merged_context': result['merged_context']
        }

    def generate_node(state: RAGState) -> RAGState:
        prompt = f"""You are an expert in industrial compressor maintenance.
Use the following context to answer the question accurately and concisely.

{state['merged_context']}

Question: {state['query']}

Answer:"""
        response = llm.invoke(prompt)
        return {**state, 'answer': response.content}

    workflow = StateGraph(RAGState)
    workflow.add_node('retrieve', retrieve_node)
    workflow.add_node('generate', generate_node)
    workflow.set_entry_point('retrieve')
    workflow.add_edge('retrieve', 'generate')
    workflow.add_edge('generate', END)
    return workflow.compile()


# ── 3. Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading graph...")
    G = load_graph()
    print("Loading vector store...")
    vector_store = load_vector_store()
    print("Building pipeline...")
    pipeline = build_pipeline(G, vector_store)

    query = "What is the role of the Elektronikon regulator?"
    print(f"\nQuery: {query}\n")

    result = pipeline.invoke({
        'query': query,
        'vector_context': '',
        'graph_context': '',
        'merged_context': '',
        'answer': ''
    })
    print("=== ANSWER ===")
    print(result['answer'])
    print("\n=== GRAPH CONTEXT USED ===")
    print(result['graph_context'])