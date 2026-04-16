from pathlib import Path
from dotenv import load_dotenv
import networkx as nx
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv(Path(__file__).parent.parent / '.env', override=True)

from graph_builder import load_graph
from graph_retriever import get_relevant_subgraph, triples_to_context

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "maintenance_manuals"


def load_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )


def hybrid_retrieve(query: str, G: nx.DiGraph, vector_store: Chroma,
                    top_k: int = 4) -> dict:
    # Vector retrieval
    vector_docs = vector_store.similarity_search(query, k=top_k)
    vector_context = "\n\n".join(doc.page_content for doc in vector_docs)

    # Graph retrieval
    triples = get_relevant_subgraph(query, G)
    graph_context = triples_to_context(triples)

    # Merged context for LLM
    merged_context = f"""VECTOR SEARCH RESULTS:
{vector_context}

KNOWLEDGE GRAPH RELATIONSHIPS:
{graph_context}"""

    return {
        "vector_context": vector_context,
        "graph_context": graph_context,
        "merged_context": merged_context
    }


if __name__ == '__main__':
    print("Loading graph...")
    G = load_graph()
    print("Loading vector store...")
    vs = load_vector_store()

    query = "What is the role of the Elektronikon regulator?"
    print(f"\nQuery: {query}\n")

    result = hybrid_retrieve(query, G, vs)
    print("=== GRAPH CONTEXT ===")
    print(result['graph_context'])
    print("\n=== VECTOR CONTEXT (first 500 chars) ===")
    print(result['vector_context'][:500])