# tests/test_questions.py
from compressor_agenticrag.agent import run

TEST_QUESTIONS = [
    {
        "query": "what is the compressor element outlet temperature shutdown level",
        "expected_route": "spec",
        "expected_keywords": ["120", "°C"],
    },
    {
        "query": "what is the nominal loading pressure for 13 bar",
        "expected_route": "spec",
        "expected_keywords": ["11.9", "bar"],
    },
    {
        "query": "what components are involved in a motor overload shutdown",
        "expected_route": "graph",
        "expected_keywords": ["motor", "overload"],
    },
    {
        "query": "how do I modify the unloading pressure setting",
        "expected_route": "vector",
        "expected_keywords": ["unloading", "pressure", "modify"],
    },
    {
        "query": "how does automatic restart after voltage failure work",
        "expected_route": "vector",
        "expected_keywords": ["voltage", "restart", "automatic"],
    },
]


def run_tests():
    print("=" * 60)
    print("Problem 3 — Agentic RAG Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, test in enumerate(TEST_QUESTIONS, 1):
        print(f"\nTest {i}: {test['query']}")
        print("-" * 60)

        result = run(test["query"])

        # Check routing
        route_correct = result["query_type"] == test["expected_route"]
        route_status = "PASS" if route_correct else "FAIL"
        print(f"  Route:    [{route_status}] expected={test['expected_route']} got={result['query_type']}")

        # Check answer contains expected keywords
        answer_lower = result["answer"].lower()
        keywords_found = [kw for kw in test["expected_keywords"] if kw.lower() in answer_lower]
        keywords_missing = [kw for kw in test["expected_keywords"] if kw.lower() not in answer_lower]
        keyword_correct = len(keywords_missing) == 0
        keyword_status = "PASS" if keyword_correct else "FAIL"
        print(f"  Keywords: [{keyword_status}] found={keywords_found} missing={keywords_missing}")

        # Print answer
        print(f"  Answer:   {result['answer']}")

        if route_correct and keyword_correct:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(TEST_QUESTIONS)} passed, {failed}/{len(TEST_QUESTIONS)} failed")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()