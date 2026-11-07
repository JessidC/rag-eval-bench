# rag-eval-bench
[![tests](https://github.com/JessidC/rag-eval-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/JessidC/rag-eval-bench/actions/workflows/tests.yml)

Measure a RAG system before you ship it. A golden set with out-of-scope questions, an LLM judge, a validator that blocks answers with made-up citations or numbers, and a paired significance test that decides whether a change ships.

At work I run a RAG assistant for an ERP support chat that handles about 1,000 conversations a day. Every change to retrieval, prompts or models goes through this kind of check before it reaches customers. This repo rebuilds the method in the open, on public documentation, so anyone can run it and read the code.

**Live report:** [jessidc.github.io/rag-eval-bench](https://jessidc.github.io/rag-eval-bench/)

## What the bench found

Corpus: the Odoo 18 Accounting and Inventory docs (176 pages, 1,022 chunks). Golden set: 60 answerable questions plus 12 that the docs cannot answer. Same model and judge in every run (`deepseek-flash`), only retrieval changes.

| Config | Accuracy | Wrong answers shown | Over-escalation | Out of scope escalated | Latency p95 | Cost / question |
|---|---|---|---|---|---|---|
| dense | 75.0% | 12.5% | 20.0% | 100% | 2.4 s | $0.0003 |
| bm25 | 87.5% | 12.1% | 3.3% | 100% | 2.0 s | $0.0003 |
| hybrid (RRF) | 87.5% | 7.3% | 8.3% | 100% | 2.5 s | $0.0003 |
| dense + rerank | 88.9% | 3.7% | 10.0% | 100% | 8.3 s | $0.0003 |
| hybrid, run again | 88.9% | 5.5% | 8.3% | 100% | 2.3 s | $0.0003 |

What the paired tests say:

- **dense → hybrid: ship.** +12.5 points, 11 questions fixed and 2 broken, McNemar p = 0.02.
- **dense → bm25: not proven.** Same +12.5 points on the headline, but it fixes 14 and breaks 5, so p = 0.06. Same average, weaker evidence, and it shows more wrong answers to users than hybrid.
- **hybrid → rerank: don't bother.** +1.4 points with p = 1.0, and p95 latency goes from 2.5 s to 8.3 s on CPU.
- **hybrid vs itself: 1.4 points of noise.** Running the same config twice moved one question. A gain smaller than that is not a gain.

Two things the bench caught in its own golden set while I built it. A question I wrote as "out of scope" (custom fields with Studio) is actually answered by the accounting docs. And the validator flagged a correct answer because the source said "eighty kilograms" and the model wrote "80". Both are fixed, and both are why a bench should be read, not just run.

## What it measures

| Metric | Meaning |
|---|---|
| Accuracy | Judge says correct, or an out-of-scope question was escalated |
| Wrong answers shown | Answers delivered to the user that were partial or wrong, over all delivered answers. The number that hurts customers. |
| Over-escalation | Answerable questions sent to a human |
| Out of scope escalated | Questions the docs cannot answer that were handed off instead of guessed |
| Source doc retrieved / MRR | Whether retrieval found the passage the question was written from |
| Cost, latency | Per question, from the provider's token counts |

## How an answer is produced

1. Retrieve 5 chunks: dense with `bge-small-en-v1.5`, BM25, or both fused with reciprocal rank fusion, with an optional cross-encoder rerank. Everything runs locally through fastembed.
2. The model answers only from the numbered sources and cites them as `[n]`, or abstains.
3. **The validator** checks the answer before it counts as delivered. Every citation must point at a retrieved source, and every number must appear in the cited text, number words included. Anything that fails is escalated, same as an abstention.
4. **The judge** compares delivered answers with the reference. Out-of-scope questions are scored by rule, since the only right outcome is to escalate.

## The gate

`rageval compare base cand --gate` exits 1 when a candidate should not ship. That happens when accuracy drops with McNemar p < 0.05, when wrong answers shown rise by more than 2 points, or when out-of-scope escalation falls.

An improvement without significance passes, labeled "not proven". Put it in CI after the eval run and a prompt change cannot quietly make things worse.

## Run it

```bash
uv sync
git clone --depth 1 --filter=blob:none --sparse --branch 18.0 https://github.com/odoo/documentation.git odoo-docs
git -C odoo-docs sparse-checkout set content/applications/finance/accounting content/applications/inventory_and_mrp/inventory

uv run rageval ingest odoo-docs/content/applications --only finance/accounting inventory_and_mrp/inventory
uv run rageval run --name dense --retrieval dense
uv run rageval run --name hybrid --retrieval hybrid
uv run rageval compare runs/dense runs/hybrid --gate
uv run rageval report runs/dense runs/hybrid --pair dense:hybrid --out docs/index.html
```

Set `DEEPSEEK_API_KEY`, or use `--provider anthropic --model <claude model>` with `ANTHROPIC_API_KEY` and pass `--price-in` and `--price-out` to get cost. `--judge-model` lets the judge be a different model from the one under test. A full run of 72 questions costs about two cents with `deepseek-flash`.

To draft a golden set for other docs, run `uv run rageval golden --n 60 --out-of-scope your_oos.jsonl`.

## Limits

- **The golden set is LLM-drafted.** The model wrote each question from one passage, phrased the way a user would ask, and drafts that leaked the passage ("according to this section") were dropped. Each line keeps `reviewed: false` until a person checks it. Questions written from a passage also make retrieval look easier than real traffic.
- **The judge is the same model as the generator.** That can favor its own phrasing. The repeat run shows the judge is stable, which is not the same as right. For decisions that matter, use a different judge model and spot-check its verdicts.
- **72 questions give wide intervals.** The bench says when a difference is not proven instead of hiding it.

## Layout

```
src/rag_eval_bench/
  corpus.py     RST to section-aware chunks
  retrieval.py  dense, BM25, RRF hybrid, cross-encoder rerank
  answer.py     grounded answering and the citation and number validator
  judge.py      LLM judge plus rule-based scoring for out-of-scope
  runner.py     runs a config over the golden set, writes results and summary
  stats.py      exact McNemar, paired bootstrap
  compare.py    paired comparison and the ship gate
  report.py     single-file HTML report
golden/         golden set and the hand-written out-of-scope questions
runs/           the results behind the table above
docs/           the published report
```

## License

Code is MIT. The golden set is derived from the [Odoo documentation](https://github.com/odoo/documentation) and shares its license, CC BY-SA 4.0.
