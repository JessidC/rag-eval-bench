import json

from rag_eval_bench.answer import Answer, numbers_in, validate
from rag_eval_bench.compare import compare
from rag_eval_bench.corpus import Chunk, chunk_document, clean_inline, parse_rst
from rag_eval_bench.judge import grade
from rag_eval_bench.retrieval import Hit, rrf
from rag_eval_bench.runner import summarize
from rag_eval_bench.stats import bootstrap_diff, mcnemar_exact

RST = """=========
Follow-ups
=========

Send reminders to customers with overdue invoices. Go to :menuselection:`Accounting -->
Configuration --> Follow-up Levels` and see :doc:`the guide <../payments>`.

.. image:: follow_up/levels.png
   :alt: levels

Configuration
=============

.. note::
   Sending letters requires ``IAP`` credits.

Each level waits a number of days before the reminder is sent, for example fifteen days.
"""


def test_parse_rst_keeps_text_drops_markup():
    sections = parse_rst(RST)
    titles = [p for p, _ in sections]
    assert ["Follow-ups"] in titles and ["Configuration"] in titles
    body = " ".join(t for _, t in sections)
    assert "Accounting > Configuration > Follow-up Levels" in body
    assert "the guide" in body and ":doc:" not in body
    assert "levels.png" not in body and "Sending letters requires IAP credits." in body


def test_chunk_document_merges_tiny_sections():
    chunks = chunk_document("acc/follow_up", RST, min_words=10)
    assert all(c.words >= 10 for c in chunks)
    assert chunks[0].id == "acc/follow_up#0"


def test_clean_inline_roles():
    assert clean_inline(":guilabel:`Save`") == "Save"
    assert clean_inline("`Odoo <https://odoo.com>`_") == "Odoo"


def test_rrf_rewards_agreement():
    assert rrf([[1, 2, 3], [2, 1, 4]])[:2] in ([1, 2], [2, 1])
    assert rrf([[5, 6], [6, 7]])[0] == 6


def _hits(*texts):
    return [Hit(Chunk(f"d#{i}", "d", "t", t), 1.0) for i, t in enumerate(texts)]


def test_validator_blocks_invented_numbers_and_bad_citations():
    hits = _hits("Reminders go out after 15 days.", "Letters need IAP credits.")
    ok = validate(Answer("Reminders go out after 15 days [1].", [1], False), hits)
    assert ok.outcome == "answered" and not ok.problems
    invented = validate(Answer("Reminders go out after 30 days [1].", [1], False), hits)
    assert invented.outcome == "escalated" and "30" in invented.problems[0]
    ghost = validate(Answer("See [7].", [7], False), hits)
    assert ghost.outcome == "escalated"
    uncited = validate(Answer("Reminders exist.", [], False), hits)
    assert "no citations" in uncited.problems


def test_validator_reads_number_words():
    hits = _hits("Create a delivery order for eighty kilograms of flour.")
    assert validate(Answer("An order for 80 kg [1].", [1], False), hits).outcome == "answered"
    assert sorted(numbers_in("twenty-one days and one hundred units")) == ["1", "100", "21"]


def test_abstention_escalates():
    assert validate(Answer("", [], True), _hits("x")).outcome == "escalated"


def test_out_of_scope_scored_by_rule_without_llm():
    assert grade(None, "q", "", "", answerable=False, outcome="escalated")["correct"] is True
    assert grade(None, "q", "", "an answer", answerable=False, outcome="answered")["correct"] is False
    assert grade(None, "q", "ref", "", answerable=True, outcome="escalated")["verdict"] == "escalated"


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(1, 9) - 0.021484375) < 1e-9
    assert mcnemar_exact(5, 6) > 0.5


def test_bootstrap_interval_contains_point():
    base = [True] * 30 + [False] * 30
    cand = [True] * 40 + [False] * 20
    point, lo, hi = bootstrap_diff(base, cand)
    assert lo <= point <= hi and abs(point - 10 / 60) < 1e-9


def _row(i, answerable, outcome, correct, verdict=None):
    return {"id": f"q{i}", "question": f"q{i}", "answerable": answerable, "outcome": outcome, "correct": correct,
            "verdict": verdict or ("correct" if correct else "incorrect"), "problems": [], "doc_hit": True,
            "chunk_rank": 1, "retrieval_s": 0.01, "answer_meta": {"cost_usd": 0.001, "latency_s": 1.0},
            "judge_meta": {}, "error": None, "answer": "a", "reason": ""}


def test_summary_counts_wrong_answers_shown():
    rows = [_row(1, True, "answered", True), _row(2, True, "answered", False), _row(3, False, "escalated", True)]
    s = summarize(rows, "x", {})
    assert s["accuracy"] == 66.7 and s["wrong_shown"] == 50.0 and s["oos_escalated"] == 100.0


def _write_run(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(summarize(rows, name, {})), encoding="utf-8")
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return d


def test_gate_blocks_more_wrong_answers_shown(tmp_path):
    base = [_row(i, True, "answered" if i < 8 else "escalated", i < 8) for i in range(10)]
    cand = [_row(i, True, "answered", i < 8) for i in range(10)]  # same accuracy, but shows 2 wrong answers
    r = compare(_write_run(tmp_path, "a", base), _write_run(tmp_path, "b", cand))
    assert r["gate"]["pass"] is False and "wrong answers shown" in r["gate"]["fails"][0]


def test_gate_passes_unproven_gain(tmp_path):
    base = [_row(i, True, "answered" if i < 5 else "escalated", i < 5) for i in range(10)]
    cand = [_row(i, True, "answered" if i < 6 else "escalated", i < 6) for i in range(10)]
    r = compare(_write_run(tmp_path, "a", base), _write_run(tmp_path, "b", cand))
    assert r["gate"]["pass"] and "not proven" in r["gate"]["verdict"]


def test_kappa_and_agreement(tmp_path):
    from rag_eval_bench.rejudge import agreement, cohen_kappa

    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0

    def write(name, verdicts):
        d = tmp_path / name
        d.mkdir()
        (d / "summary.json").write_text(json.dumps({"name": name, "config": {}}))
        rows = [{"id": f"q{i}", "question": "q", "verdict": v, "reason": ""} for i, v in enumerate(verdicts)]
        (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        return d

    a = write("a", ["correct", "correct", "partial", "escalated"])
    b = write("b", ["correct", "incorrect", "partial", "escalated"])
    r = agreement(a, b)
    assert r["n"] == 3  # the rule-scored escalation is left out
    assert r["a_only_correct"] == 1 and r["b_only_correct"] == 0


def test_rejudge_keeps_answers_and_changes_only_verdicts(tmp_path):
    from rag_eval_bench.rejudge import rejudge

    class Strict:
        provider, model = "fake", "strict"

        def complete_json(self, system, user, max_tokens=300):
            return {"verdict": "partial", "reason": "misses a step"}, {"cost_usd": 0.0}

    run = tmp_path / "dense"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"name": "dense", "config": {"retrieval": "dense"}}))
    rows = [
        {"id": "q1", "question": "How?", "answerable": True, "outcome": "answered", "answer": "Do X [1].",
         "verdict": "correct", "correct": True, "reason": "", "problems": [], "doc_hit": True, "chunk_rank": 1,
         "retrieval_s": 0.1, "answer_meta": {"latency_s": 1.0}, "judge_meta": {}, "error": None},
        {"id": "q2", "question": "Weather?", "answerable": False, "outcome": "escalated", "answer": None,
         "verdict": "correct", "correct": True, "reason": "", "problems": [], "doc_hit": None, "chunk_rank": None,
         "retrieval_s": 0.1, "answer_meta": {"latency_s": 1.0}, "judge_meta": {}, "error": None},
    ]
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    golden = [{"id": "q1", "reference": "Do X then Y."}, {"id": "q2"}]
    s = rejudge(run, golden, Strict(), tmp_path / "out")
    new = [json.loads(line) for line in (tmp_path / "out" / "dense" / "results.jsonl").read_text().splitlines()]
    assert new[0]["answer"] == "Do X [1]." and new[0]["verdict"] == "partial" and not new[0]["correct"]
    assert new[1]["verdict"] == "correct"  # out of scope stays scored by rule
    assert s["config"]["judge"] == "fake/strict" and s["accuracy"] == 50.0
