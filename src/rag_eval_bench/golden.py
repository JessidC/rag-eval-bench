"""Draft a golden set from the corpus.

An LLM reads one passage at a time and writes the question a real user would ask that the
passage answers, plus a short reference answer. Drafts that lean on the wording of the
passage ("according to this section") or that are too vague are dropped. The result is a
starting point: review it before trusting the numbers (see `reviewed` in each line).

Out-of-scope questions live in a separate hand-written file and are merged in, so the
bench also measures whether the assistant knows when to hand off.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .corpus import Chunk

SYSTEM = """You write evaluation questions for a support assistant that answers Odoo users from the documentation.
Given one documentation passage, write ONE question that a real Odoo user would type into a support chat and that
this passage fully answers. Then write the reference answer from the passage in 1 to 3 sentences, with the menu path
or steps if the passage gives them.
Rules: the question must make sense without seeing the passage (never "in this section", "according to the text");
no yes/no questions; ask about how to do something, where a setting is, or what a feature does.
Phrase it the way a user describes their problem, in their own words: do not copy the passage's headings, menu
names or exact terms into the question (the reference answer can use them).
If the passage is only an index, a list of links or too thin to support a real question, return {"skip": true}.
Reply with JSON only: {"question": "...", "reference": "..."} or {"skip": true}"""

LEAKY = re.compile(r"\b(this|the) (section|passage|text|document|article)\b|according to", re.I)


def sample_passages(chunks: list[Chunk], n: int, seed: int = 11, min_words: int = 70) -> list[Chunk]:
    """Spread the sample across documents so one long page does not dominate."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        if c.words >= min_words:
            by_doc[c.doc].append(c)
    rng = random.Random(seed)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    picked: list[Chunk] = []
    while len(picked) < n and any(by_doc.values()):
        for d in docs:
            if by_doc[d] and len(picked) < n:
                picked.append(by_doc[d].pop(rng.randrange(len(by_doc[d]))))
    return picked


def draft(llm, chunks: list[Chunk], n: int, seed: int = 11) -> list[dict]:
    out = []
    for c in sample_passages(chunks, int(n * 1.4), seed):
        if len(out) >= n:
            break
        try:
            data, _ = llm.complete_json(SYSTEM, f"Passage title: {c.title}\n\n{c.text}", max_tokens=400)
        except Exception:
            continue
        q, ref = str(data.get("question", "")).strip(), str(data.get("reference", "")).strip()
        if data.get("skip") or not q or not ref or LEAKY.search(q) or len(q.split()) < 5:
            continue
        out.append({"id": f"q{len(out) + 1:03d}", "question": q, "reference": ref, "answerable": True,
                    "source_doc": c.doc, "source_chunk": c.id, "reviewed": False})
    return out


def merge_out_of_scope(items: list[dict], oos_path: Path) -> list[dict]:
    with open(oos_path, encoding="utf-8") as f:
        oos = [json.loads(line) for line in f if line.strip()]
    for i, o in enumerate(oos, 1):
        items.append({"id": f"oos{i:02d}", "question": o["question"], "reference": "", "answerable": False,
                      "source_doc": None, "source_chunk": None, "reviewed": True})
    return items
