"""
judge.py — LLM-as-judge evaluation node. Carried over from v5.5 unchanged.
"""

from __future__ import annotations

import json

from compressor_longtermrag.nodes.clients import classifier_llm


def judge_node(state) -> dict:
    prompt = f"""You are evaluating whether an answer fully addresses a maintenance question.

Question: {state.query}
Context retrieved: {state.context}
Answer generated: {state.answer}

Evaluate whether the answer:
1. Directly addresses the question asked
2. Contains specific values or steps where the question requires them
3. Is grounded in the provided context — no hallucination

Return a JSON object with exactly three fields:
  "verdict": "good" or "insufficient"
  "score": a float between 0.0 and 1.0 indicating answer completeness
  "evaluation": a one-sentence description of what is missing, or "" if verdict is good

Return ONLY the JSON object — no explanation, no markdown fences.
"""
    result = classifier_llm.invoke(prompt)

    judge_parse_error = False
    try:
        raw = result.content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "insufficient").lower()
        score = float(parsed.get("score", 0.0))
        evaluation = parsed.get("evaluation", "")
        if verdict not in ("good", "insufficient"):
            verdict = "insufficient"
    except Exception:
        judge_parse_error = True
        verdict = "insufficient"
        score = 0.0
        evaluation = "Judge response could not be parsed."

    new_attempt = state.attempt + 1
    new_score_trend = state.score_trend + [score]

    return {
        "verdict": verdict,
        "score": score,
        "evaluation": evaluation,
        "attempt": new_attempt,
        "score_trend": new_score_trend,
        "judge_parse_error": judge_parse_error,
    }