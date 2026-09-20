"""Run one retrieval/answer configuration over the golden set and summarize it."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answer import answer
from .judge import grade
from .stats import percentile


def read_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_one(item: dict, retriever, llm, judge_llm) -> dict:
    t0 = time.perf_counter()
    hits = retriever.search(item["question"])
    t_retrieval = time.perf_counter() - t0
    docs = [h.chunk.doc for h in hits]
    ids = [h.chunk.id for h in hits]
    src_chunk, src_doc = item.get("source_chunk"), item.get("source_doc")
    rank = ids.index(src_chunk) + 1 if src_chunk in ids else None
    try:
        ans = answer(llm, item["question"], hits)
        g = grade(judge_llm, item["question"], item.get("reference", ""), ans.text, item["answerable"], ans.outcome)
        error = None
    except Exception as e:  # one bad call should not sink the run; it is counted as a miss
        ans, g, error = None, {"verdict": "error", "correct": False, "reason": str(e)[:200], "meta": {}}, str(e)[:200]
    return {
        "id": item["id"], "question": item["question"], "answerable": item["answerable"],
        "retrieved": ids, "doc_hit": src_doc in docs if src_doc else None, "chunk_rank": rank,
        "answer": ans.text if ans else None, "outcome": ans.outcome if ans else "error",
        "problems": ans.problems if ans else [], "verdict": g["verdict"], "correct": g["correct"], "reason": g["reason"],
        "retrieval_s": round(t_retrieval, 4), "answer_meta": ans.meta if ans else {}, "judge_meta": g.get("meta", {}),
        "error": error,
    }


def summarize(rows: list[dict], name: str, config: dict) -> dict:
    ans_rows = [r for r in rows if r["answerable"]]
    oos = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if r["outcome"] == "answered"]
    ranks = [r["chunk_rank"] for r in ans_rows]
    costs = [(r["answer_meta"].get("cost_usd") or 0) + (r["judge_meta"].get("cost_usd") or 0) for r in rows]
    lat = [r["answer_meta"].get("latency_s", 0) + r["retrieval_s"] for r in rows if r["answer_meta"]]
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "name": name, "config": config, "questions": len(rows), "answerable": len(ans_rows), "out_of_scope": len(oos),
        "accuracy": pct(sum(r["correct"] for r in rows), len(rows)),
        "answerable_accuracy": pct(sum(r["correct"] for r in ans_rows), len(ans_rows)),
        "partial": sum(r["verdict"] == "partial" for r in rows),
        "wrong_shown": pct(sum(r["outcome"] == "answered" and not r["correct"] for r in rows), len(answered)),
        "over_escalation": pct(sum(r["outcome"] == "escalated" for r in ans_rows), len(ans_rows)),
        "oos_escalated": pct(sum(r["outcome"] == "escalated" for r in oos), len(oos)),
        "validator_blocks": sum(1 for r in rows if r["problems"] and r["problems"] != ["model abstained"]),
        "doc_hit": pct(sum(bool(r["doc_hit"]) for r in ans_rows), len(ans_rows)),
        "chunk_hit": pct(sum(x is not None for x in ranks), len(ranks)),
        "mrr": round(sum(1 / x for x in ranks if x) / len(ranks), 3) if ranks else None,
        "cost_usd": round(sum(costs), 4), "cost_per_q_usd": round(sum(costs) / len(rows), 5) if rows else None,
        "latency_p50_s": round(percentile(lat, 0.5), 2), "latency_p95_s": round(percentile(lat, 0.95), 2),
        "errors": sum(1 for r in rows if r["error"]),
    }


def run(golden: list[dict], retriever, llm, judge_llm, out_dir: Path, name: str, workers: int = 4) -> dict:
    out_dir = Path(out_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(lambda it: run_one(it, retriever, llm, judge_llm), golden))
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    config = {"retrieval": retriever.name, "model": f"{llm.provider}/{llm.model}", "judge": f"{judge_llm.provider}/{judge_llm.model}"}
    summary = summarize(rows, name, config)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_run(path: Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with open(path / "results.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return summary, rows
