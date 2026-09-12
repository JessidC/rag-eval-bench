"""Grounded answering with a validator in front of the user.

The model must cite the numbered sources it used, or abstain. Before an answer counts as
delivered, code checks it: every citation must point at a source that was actually
retrieved, and every number in the answer must appear in the cited text. Anything that
fails, or any abstention, is escalated to a human instead of shown.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import Hit

SYSTEM = """You are the support assistant for Odoo users. Answer ONLY from the numbered sources.
- Cite every claim with the source number in brackets, like [2].
- Give the menu path or steps when the sources do.
- If the sources do not contain the answer, do not guess: set "abstain" to true.
Reply with JSON only: {"answer": "...", "citations": [numbers], "abstain": true|false}"""

NUMBER = re.compile(r"(?<![\w\[])\d+(?:[.,]\d+)?%?(?![\w\]])")
UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
WORD_NUMBER = re.compile(r"\b(" + "|".join(TENS) + r")(?:[- ](" + "|".join(UNITS[1:10]) + r"))?\b|\b(" + "|".join(UNITS) + r"|hundred)\b", re.I)


def numbers_in(text: str) -> set[str]:
    """Numbers a source states, as digits or as words ("eighty kilograms" -> 80)."""
    found = {n.rstrip("%").replace(",", ".") for n in NUMBER.findall(text)}
    for m in WORD_NUMBER.finditer(text):
        if m.group(1):
            found.add(str(TENS[m.group(1).lower()] + (UNITS.index(m.group(2).lower()) if m.group(2) else 0)))
        else:
            w = m.group(3).lower()
            found.add("100" if w == "hundred" else str(UNITS.index(w)))
    return found


def format_sources(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{n}] {h.chunk.title}\n{h.chunk.text}" for n, h in enumerate(hits, 1))


@dataclass
class Answer:
    text: str
    citations: list[int]
    abstained: bool
    outcome: str = "answered"  # answered | escalated
    problems: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def validate(ans: Answer, hits: list[Hit]) -> Answer:
    if ans.abstained:
        ans.outcome = "escalated"
        ans.problems.append("model abstained")
        return ans
    valid = set(range(1, len(hits) + 1))
    cited_in_text = {int(n) for n in re.findall(r"\[(\d+)\]", ans.text)}
    cites = set(ans.citations) | cited_in_text
    if not cites:
        ans.problems.append("no citations")
    bad = sorted(cites - valid)
    if bad:
        ans.problems.append(f"cites sources that were not retrieved: {bad}")
    stated = numbers_in(" ".join(hits[n - 1].chunk.text for n in cites & valid))
    plain = re.sub(r"\[\d+\]", "", ans.text)
    for num in NUMBER.findall(plain):
        if num.rstrip("%").replace(",", ".") not in stated:
            ans.problems.append(f"number not in cited sources: {num}")
    ans.outcome = "escalated" if ans.problems else "answered"
    return ans


def answer(llm, question: str, hits: list[Hit]) -> Answer:
    user = f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"
    data, meta = llm.complete_json(SYSTEM, user, max_tokens=700)
    cites = [int(c) for c in data.get("citations", []) if str(c).strip().isdigit()]
    ans = Answer(text=str(data.get("answer", "")).strip(), citations=cites, abstained=bool(data.get("abstain")), meta=meta)
    return validate(ans, hits)
