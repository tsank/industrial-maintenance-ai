import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'compressor_graphrag'))

from graph_builder import load_graph
from hybrid_retriever import load_vector_store
from pipeline import build_pipeline

TEST_QUESTIONS = [
    "What is the role of the Elektronikon regulator?",
    "What should I do if the compressor overheats?",
    "How do I perform routine maintenance on the GA5?",
    "What are the safety warnings before starting the compressor?",
    "How does the automatic restart function work?",
]


def run_evaluation():
    print("Loading graph...")
    G = load_graph()
    print("Loading vector store...")
    vector_store = load_vector_store()
    print("Building pipeline...")
    pipeline = build_pipeline(G, vector_store)
    print("\n" + "="*60)

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 60)

        initial_state = {
            'query': question,
            'vector_context': '',
            'graph_context': '',
            'merged_context': '',
            'answer': ''
        }

        result = pipeline.invoke(initial_state)

        print(f"ANSWER:\n{result['answer']}")
        print(f"\nGRAPH CONTEXT USED:")
        print(result['graph_context'])
        print("=" * 60)


if __name__ == '__main__':
    run_evaluation()