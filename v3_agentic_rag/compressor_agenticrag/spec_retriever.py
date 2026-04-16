# compressor_agenticrag/spec_retriever.py
import json
import re
from pathlib import Path

SPECS_PATH = Path(__file__).parent.parent / "data" / "specs.json"


def load_specs() -> dict:
    with open(SPECS_PATH) as f:
        return json.load(f)


def tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, drop short tokens."""
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2}


def score(query_tokens: set[str], key: str, value: str) -> int:
    """Count how many query tokens appear in the key or value."""
    target_tokens = tokenize(key) | tokenize(value)
    return len(query_tokens & target_tokens)


def retrieve_specs(query: str, top_k: int = 3) -> list[dict]:
    """
    Match query against spec keys and values by token overlap.
    Returns top_k matches as list of {key, value, score} dicts.
    """
    specs = load_specs()
    query_tokens = tokenize(query)

    scored = [
        {"key": k, "value": v, "score": score(query_tokens, k, v)}
        for k, v in specs.items()
    ]

    # Filter out zero-score matches, sort descending
    matches = [s for s in scored if s["score"] > 0]
    matches.sort(key=lambda x: x["score"], reverse=True)

    return matches[:top_k]


def format_spec_context(matches: list[dict]) -> str:
    """Format matches as readable context for the LLM."""
    if not matches:
        return "No matching specifications found."
    lines = [f"- {m['key']}: {m['value']}" for m in matches]
    return "Relevant specifications:\n" + "\n".join(lines)


if __name__ == "__main__":
    test_queries = [
        "what is the maximum unloading pressure",
        "shutdown temperature",
        "service interval hours",
        "motor power voltage",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        matches = retrieve_specs(q)
        print(format_spec_context(matches))