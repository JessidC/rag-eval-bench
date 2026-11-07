"""Grade a finished run again with a different judge, and measure how often two judges agree.

Re-judging keeps everything else fixed: the same retrieved chunks, the same delivered answers.
Only the verdicts change, so any shift in a metric or a comparison comes from the judge.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .judge import grade
from .runner import load_run, summarize


def rejudge_row(row: dict, golden_by_id: dict, judge_llm) -> dict:
    item = golden_by_id[row["id"]]
    new = dict(row)
    if row["outcome"] == "error":
        return new
    try:
        g = grade(judge_llm, row["question"], item.get("reference", ""), row["answer"] or "", row["answerable"],
                  row["outcome"])
        new.update(verdict=g["verdict"], correct=g["correct"], reason=g["reason"], judge_meta=g.get("meta", {}),
                   error=None)
    except Exception as e:  # counted as a miss, same as in a normal run
        new.update(verdict="error", correct=False, reason=str(e)[:200], judge_meta={}, error=str(e)[:200])
    return new


def rejudge(run_dir: Path, golden: list[dict], judge_llm, out_dir: Path, workers: int = 4) -> dict:
    summary, rows = load_run(run_dir)
    golden_by_id = {g["id"]: g for g in golden}
    with ThreadPoolExecutor(workers) as pool:
        new_rows = list(pool.map(lambda r: rejudge_row(r, golden_by_id, judge_llm), rows))
    out = Path(out_dir) / summary["name"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    config = dict(summary["config"], judge=f"{judge_llm.provider}/{judge_llm.model}", rejudged_from=str(run_dir))
    new_summary = summarize(new_rows, summary["name"], config)
    (out / "summary.json").write_text(json.dumps(new_summary, indent=2), encoding="utf-8")
    return new_summary


def cohen_kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def agreement(run_a: Path, run_b: Path) -> dict:
    """Agreement on the verdicts an LLM gave in both runs (rule-scored rows are left out)."""
    _, ra = load_run(run_a)
    _, rb = load_run(run_b)
    by_id = {r["id"]: r for r in rb}
    llm_verdicts = ("correct", "partial", "incorrect")
    pairs = [(x, by_id[x["id"]]) for x in ra
             if x["id"] in by_id and x["verdict"] in llm_verdicts and by_id[x["id"]]["verdict"] in llm_verdicts]
    va = [x["verdict"] for x, _ in pairs]
    vb = [y["verdict"] for _, y in pairs]
    ca = [v == "correct" for v in va]
    cb = [v == "correct" for v in vb]
    n = len(pairs)
    return {
        "n": n,
        "agree_correct": round(100 * sum(x == y for x, y in zip(ca, cb)) / n, 1) if n else None,
        "kappa_correct": round(cohen_kappa(ca, cb), 3) if n else None,
        "agree_verdict": round(100 * sum(x == y for x, y in zip(va, vb)) / n, 1) if n else None,
        "kappa_verdict": round(cohen_kappa(va, vb), 3) if n else None,
        "a_only_correct": sum(x and not y for x, y in zip(ca, cb)),
        "b_only_correct": sum(y and not x for x, y in zip(ca, cb)),
        "disagreements": [{"id": x["id"], "question": x["question"], "a": x["verdict"], "b": y["verdict"],
                           "b_reason": y["reason"]} for x, y in pairs if x["verdict"] != y["verdict"]],
    }
