"""Compare a candidate run against a baseline on the same golden set, and decide if it ships.

Gate rules (a release is blocked if any fails):
1. Accuracy may not drop with significance (exact McNemar, p < 0.05).
2. Wrong answers shown to users may not rise by more than 2 points: a wrong answer
   delivered with confidence costs more than an escalation.
3. Out-of-scope questions must keep being escalated (no drop).
An improvement that is not significant passes, but is reported as "not proven".
"""
from __future__ import annotations

from pathlib import Path

from .runner import load_run
from .stats import bootstrap_diff, mcnemar_exact, paired_counts

METRICS = [
    ("accuracy", "Accuracy", "%", +1), ("answerable_accuracy", "Accuracy on answerable", "%", +1),
    ("wrong_shown", "Wrong answers shown", "%", -1), ("over_escalation", "Over-escalation", "%", -1),
    ("oos_escalated", "Out-of-scope escalated", "%", +1), ("doc_hit", "Source doc retrieved", "%", +1),
    ("mrr", "MRR (source chunk)", "", +1), ("cost_per_q_usd", "Cost per question", "USD", -1),
    ("latency_p95_s", "Latency p95", "s", -1),
]


def compare(base_dir: Path, cand_dir: Path) -> dict:
    sb, rb = load_run(base_dir)
    sc, rc = load_run(cand_dir)
    by_id = {r["id"]: r for r in rc}
    pairs = [(r, by_id[r["id"]]) for r in rb if r["id"] in by_id]
    base = [x["correct"] for x, _ in pairs]
    cand = [y["correct"] for _, y in pairs]
    b, c = paired_counts(base, cand)
    p = mcnemar_exact(b, c)
    diff, lo, hi = bootstrap_diff(base, cand) if pairs else (0.0, 0.0, 0.0)
    fails = []
    if c < b and p < 0.05:
        fails.append(f"accuracy dropped significantly ({b} lost vs {c} gained, p={p:.4f})")
    if (sc.get("wrong_shown") or 0) - (sb.get("wrong_shown") or 0) > 2.0:
        fails.append(f"wrong answers shown rose from {sb['wrong_shown']}% to {sc['wrong_shown']}%")
    if sb.get("oos_escalated") is not None and (sc.get("oos_escalated") or 0) < sb["oos_escalated"]:
        fails.append(f"out-of-scope escalation fell from {sb['oos_escalated']}% to {sc['oos_escalated']}%")
    if fails:
        verdict = "block"
    elif c > b and p < 0.05:
        verdict = "ship: significant improvement"
    elif c > b:
        verdict = "ship allowed: improvement not proven"
    else:
        verdict = "ship allowed: no quality change"
    flips = [{"id": x["id"], "question": x["question"], "baseline": x["verdict"], "candidate": y["verdict"],
              "candidate_answer": y["answer"], "reason": y["reason"]}
             for x, y in pairs if x["correct"] != y["correct"]]
    return {"baseline": sb, "candidate": sc, "paired": {"n": len(pairs), "only_baseline": b, "only_candidate": c,
            "p_value": p, "diff": diff, "ci95": [lo, hi]}, "gate": {"pass": not fails, "verdict": verdict, "fails": fails},
            "flips": flips}


def render_text(r: dict) -> str:
    sb, sc, pr = r["baseline"], r["candidate"], r["paired"]
    lines = [f"{'metric':<26}{sb['name']:>16}{sc['name']:>16}"]
    for key, label, unit, _ in METRICS:
        lines.append(f"{label:<26}{str(sb.get(key)):>16}{str(sc.get(key)):>16}  {unit}")
    lines += ["", f"paired over {pr['n']} questions: baseline-only right {pr['only_baseline']}, candidate-only right "
              f"{pr['only_candidate']}, McNemar p={pr['p_value']:.4f}",
              f"accuracy difference {100 * pr['diff']:+.1f} pts (95% CI {100 * pr['ci95'][0]:+.1f} to {100 * pr['ci95'][1]:+.1f})",
              f"gate: {r['gate']['verdict'].upper()}"]
    lines += [f"  - {f}" for f in r["gate"]["fails"]]
    return "\n".join(lines)
