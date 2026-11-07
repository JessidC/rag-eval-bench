"""Scoring one question.

Out-of-scope questions are scored in code: the only right outcome is to escalate. For
answerable questions an LLM judge compares the delivered answer with the reference; an
escalated answer is scored as a miss and counted separately as over-escalation.
"""
from __future__ import annotations

SYSTEM = """You grade answers from a support assistant against a reference answer written from the documentation.
Verdicts:
- "correct": the answer contains the key facts or steps of the reference and nothing that contradicts it.
- "partial": right direction but misses a key step or fact a user would need.
- "incorrect": wrong, contradicts the reference, or answers a different question.
Extra correct detail is fine. Wording does not matter.
Reply with JSON only: {"reason": "one sentence", "verdict": "correct|partial|incorrect"}"""


def grade(llm, question: str, reference: str, answer_text: str, answerable: bool, outcome: str) -> dict:
    if not answerable:
        ok = outcome == "escalated"
        return {"verdict": "correct" if ok else "incorrect", "correct": ok,
                "reason": "out of scope: escalated" if ok else "out of scope but answered", "judged_by": "rule", "meta": {}}
    if outcome == "escalated":
        return {"verdict": "escalated", "correct": False, "reason": "escalated an answerable question",
                "judged_by": "rule", "meta": {}}
    user = f"Question: {question}\n\nReference answer: {reference}\n\nAssistant answer: {answer_text}"
    try:
        data, meta = llm.complete_json(SYSTEM, user, max_tokens=300)
    except ValueError:  # malformed JSON happens now and then; one retry before counting a miss
        data, meta = llm.complete_json(SYSTEM, user, max_tokens=300)
    verdict = str(data.get("verdict", "")).lower().strip()
    if verdict not in ("correct", "partial", "incorrect"):
        verdict = "incorrect"
    return {"verdict": verdict, "correct": verdict == "correct", "reason": str(data.get("reason", ""))[:300],
            "judged_by": "llm", "meta": meta}
