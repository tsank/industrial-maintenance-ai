# Equivalent Manual RAG - Interactive Q&A Application
# CLI interface for technicians to query equipment manuals
# Type 'exit' to quit, 'help' for usage tips

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from chain import build_rag_chain

HELP_TEXT = """
Usage tips:
- Ask specific questions about equipment maintenance, safety, troubleshooting, etc.
- Include equipment name if known
- Ask about procedures, torque specs, safety precautions, etc.
- Type 'exit' to quit the application
- Type 'help' to see this message again
"""

def main():
    print("=" * 50)
    print("Welcome to the Equipment Manual Q&A System")
    print("=" * 50)
    print(HELP_TEXT)

    print("Connecting to knowledge base...", end="", flush=True)
    try:
        chain = build_rag_chain()
        print(" CReady\n")
    except Exception as e:
        print(f"\nError connecting to knowledge base: {e}")
        print("Make sure ChromaDB is runnng and documets are ingested.")
        sys.exit(1)

    while True:
        try:
            query = input("Technician query: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye. Exiting application.")
            break

        if not query:
            continue
        if query.lower() == "exit":
            print("Goodbye. Exiting application.")
            break
        if query.lower() == "help":
            print(HELP_TEXT)
            continue

        print("Searching manuals...", end="", flush=True)
        try:
            response = chain.invoke(query)
            print("\n")
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"\nError processing query: {e}")

        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()