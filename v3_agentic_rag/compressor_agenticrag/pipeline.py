# compressor_agenticrag/pipeline.py
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from compressor_agenticrag.agent import run


def main():
    print("Compressor Agentic RAG — Problem 3")
    print("Type 'quit' to exit\n")

    while True:
        query = input("Query: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            break

        print()
        result = run(query)
        print(f"\nAnswer: {result['answer']}")
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()