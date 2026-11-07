"""rageval: ingest, golden, run, rejudge, agreement, compare, gate, report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _llm(args, prefix=""):
    from .llm import LLM

    provider = getattr(args, f"{prefix}provider")
    model = getattr(args, f"{prefix}model")
    price = (args.price_in, args.price_out) if args.price_in is not None and args.price_out is not None else None
    return LLM(provider, model, price)


def cmd_ingest(args):
    from .corpus import load_corpus
    from .retrieval import build_index

    chunks = load_corpus(Path(args.docs), args.only)
    build_index(chunks, Path(args.index))
    print(f"indexed {len(chunks)} chunks from {args.docs} into {args.index}")


def cmd_golden(args):
    from .corpus import read_chunks
    from .golden import draft, merge_out_of_scope

    chunks = read_chunks(Path(args.index) / "chunks.jsonl")
    items = draft(_llm(args), chunks, args.n, args.seed)
    if args.out_of_scope:
        items = merge_out_of_scope(items, Path(args.out_of_scope))
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} questions to {args.out}")


def cmd_run(args):
    from .retrieval import Retriever
    from .runner import read_golden, run

    golden = read_golden(Path(args.golden))
    if args.limit:
        golden = golden[: args.limit]
    retriever = Retriever(Path(args.index), args.retrieval, args.k, args.rerank)
    llm = _llm(args)
    judge = _llm(args, "judge_") if args.judge_model else llm
    s = run(golden, retriever, llm, judge, Path(args.out), args.name, args.workers)
    print(json.dumps(s, indent=2))


def cmd_rejudge(args):
    from .rejudge import rejudge
    from .runner import read_golden

    golden = read_golden(Path(args.golden))
    judge = _llm(args, "judge_")
    for run_dir in args.runs:
        s = rejudge(Path(run_dir), golden, judge, Path(args.out), args.workers)
        print(f"{s['name']}: accuracy {s['accuracy']}%, wrong shown {s['wrong_shown']}%, errors {s['errors']}")


def cmd_agreement(args):
    from .rejudge import agreement

    r = agreement(Path(args.a), Path(args.b))
    print(f"{r['n']} answers graded by an LLM in both runs")
    print(f"correct vs not: {r['agree_correct']}% agree, kappa {r['kappa_correct']} "
          f"(only A says correct: {r['a_only_correct']}, only B: {r['b_only_correct']})")
    print(f"three verdicts: {r['agree_verdict']}% agree, kappa {r['kappa_verdict']}")
    if args.show:
        for d in r["disagreements"]:
            print(f"  {d['id']}: {d['a']} -> {d['b']}  {d['b_reason']}")


def cmd_compare(args):
    from .compare import compare, render_text

    result = compare(Path(args.baseline), Path(args.candidate))
    print(render_text(result))
    if args.gate:
        sys.exit(0 if result["gate"]["pass"] else 1)


def cmd_report(args):
    from .report import write_report

    pairs = [tuple(p.split(":", 1)) for p in args.pair] if args.pair else None
    path = write_report([Path(p) for p in args.runs], Path(args.out), baseline=args.baseline, pairs=pairs,
                        title=args.title, notes=args.notes, second_judge=args.second_judge,
                        limits=args.limit_note, source=args.source)
    print(f"report written to {path}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rageval", description="Measure a RAG system before you ship it.")
    sub = p.add_subparsers(required=True)

    def llm_flags(sp, judge=False):
        sp.add_argument("--provider", default="deepseek", choices=["deepseek", "anthropic", "orca"])
        sp.add_argument("--model", default="deepseek-flash")
        sp.add_argument("--price-in", type=float, help="USD per million input tokens")
        sp.add_argument("--price-out", type=float, help="USD per million output tokens")
        if judge:
            sp.add_argument("--judge-provider", default="deepseek", choices=["deepseek", "anthropic", "orca"])
            sp.add_argument("--judge-model", help="defaults to the model under test")

    s = sub.add_parser("ingest", help="chunk the docs and build the index")
    s.add_argument("docs")
    s.add_argument("--index", default=".index")
    s.add_argument("--only", nargs="+", help="keep only docs under these path prefixes")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("golden", help="draft a golden set from the index")
    s.add_argument("--index", default=".index")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--out-of-scope", help="JSONL of questions the docs cannot answer")
    s.add_argument("--out", default="golden/golden.jsonl")
    llm_flags(s)
    s.set_defaults(fn=cmd_golden)

    s = sub.add_parser("run", help="run one configuration over the golden set")
    s.add_argument("--golden", default="golden/golden.jsonl")
    s.add_argument("--index", default=".index")
    s.add_argument("--retrieval", default="dense", choices=["dense", "bm25", "hybrid"])
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--name", required=True)
    s.add_argument("--out", default="runs")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--limit", type=int)
    llm_flags(s, judge=True)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("rejudge", help="grade finished runs again with another judge; answers stay the same")
    s.add_argument("runs", nargs="+")
    s.add_argument("--golden", default="golden/golden.jsonl")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--judge-provider", required=True, choices=["deepseek", "anthropic", "orca"])
    s.add_argument("--judge-model", required=True)
    s.add_argument("--price-in", type=float, help="USD per million input tokens")
    s.add_argument("--price-out", type=float, help="USD per million output tokens")
    s.set_defaults(fn=cmd_rejudge)

    s = sub.add_parser("agreement", help="how often two runs' judges agree on the same answers")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--show", action="store_true", help="list each disagreement")
    s.set_defaults(fn=cmd_agreement)

    s = sub.add_parser("compare", help="paired comparison of two runs")
    s.add_argument("baseline")
    s.add_argument("candidate")
    s.add_argument("--gate", action="store_true", help="exit 1 if the candidate should not ship")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("report", help="static HTML report of one or more runs")
    s.add_argument("runs", nargs="+")
    s.add_argument("--baseline", help="run name to compare the others against")
    s.add_argument("--pair", action="append", help="baseline:candidate run names to compare (repeatable)")
    s.add_argument("--title", default="RAG evaluation report")
    s.add_argument("--notes", default="")
    s.add_argument("--second-judge", help="folder with the same runs graded by another judge (from rageval rejudge)")
    s.add_argument("--limit-note", action="append", help="a known limit of this evaluation, shown in the report (repeatable)")
    s.add_argument("--source", default="", help="link to the code and data behind the report")
    s.add_argument("--out", default="report/index.html")
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
