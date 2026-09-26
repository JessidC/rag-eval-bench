"""Static, single-file HTML report: every run side by side, then each paired comparison
with its gate verdict and the questions that flipped. No JS framework, no network."""
from __future__ import annotations

import html
from pathlib import Path

from .compare import METRICS, compare
from .runner import load_run

CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#14171f;--muted:#5d6472;--line:#dfe3ea;--accent:#2f5bd3;--ok:#1d7a4a;--okbg:#e2f3ea;
--warn:#8a5a00;--warnbg:#fbf0d9;--bad:#a3392f;--badbg:#f8e4e1;--bar:#c9d4f2;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--panel:#171b23;--ink:#e6e9ef;--muted:#9aa2b1;--line:#2a303c;--accent:#8aa6ff;
--ok:#6fd39c;--okbg:#16301f;--warn:#f0c060;--warnbg:#352a12;--bad:#f29a8f;--badbg:#3a1f1c;--bar:#2c3a63}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
main{max-width:1040px;margin:0 auto;padding:32px max(16px,env(safe-area-inset-left)) 64px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:18px;margin:36px 0 12px}p{margin:0 0 10px}.muted{color:var(--muted)}
.meta{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--muted);margin:10px 0 0}
.meta b{color:var(--ink);font-weight:600}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
tbody tr:last-child td{border-bottom:0}td.best{font-weight:700;color:var(--accent)}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--bar);vertical-align:middle;margin-left:8px}
.cmp{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:0 0 14px}
.cmp h3{margin:0 0 6px;font-size:16px;display:flex;flex-wrap:wrap;align-items:center;gap:10px}
.pill{font-size:12px;font-weight:700;padding:3px 9px;border-radius:999px;letter-spacing:.02em}
.pill.ship{background:var(--okbg);color:var(--ok)}.pill.hold{background:var(--warnbg);color:var(--warn)}.pill.block{background:var(--badbg);color:var(--bad)}
.stat{font-family:var(--mono);font-size:13px;color:var(--muted)}
details{margin-top:8px}summary{cursor:pointer;color:var(--accent);font-size:14px}
ul.flips{margin:8px 0 0;padding-left:18px;font-size:14px}ul.flips li{margin:0 0 8px}
.tag{font-family:var(--mono);font-size:12px;padding:1px 6px;border-radius:4px;background:var(--bg);border:1px solid var(--line)}
footer{margin-top:40px;font-size:13px;color:var(--muted)}
"""


def _fmt(v, unit):
    if v is None:
        return "–"
    if unit == "USD":
        return f"${v:.5f}"
    if unit == "s":
        return f"{v:.2f} s"
    if unit == "%":
        return f"{v:.1f}%"
    return f"{v}"


def _pill(verdict: str) -> str:
    cls = "block" if verdict == "block" else "ship" if "significant" in verdict else "hold"
    return f'<span class="pill {cls}">{html.escape(verdict.upper())}</span>'


def write_report(run_dirs: list[Path], out: Path, baseline: str | None = None, pairs: list[tuple[str, str]] | None = None,
                 title: str = "RAG evaluation report", notes: str = "") -> Path:
    runs = {}
    for d in run_dirs:
        s, rows = load_run(d)
        runs[s["name"]] = (d, s, rows)
    names = list(runs)
    if not pairs and baseline:
        pairs = [(baseline, n) for n in names if n != baseline]
    pairs = pairs or []
    first = runs[names[0]][1]

    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = []
    for key, label, unit, better in METRICS:
        vals = [runs[n][1].get(key) for n in names]
        nums = [v for v in vals if isinstance(v, (int, float))]
        best = (max(nums) if better > 0 else min(nums)) if nums and len(set(nums)) > 1 else None
        top = max(nums) if nums else 0
        cells = []
        for v in vals:
            bar = f'<span class="bar" style="width:{max(2, 60 * v / top):.0f}px"></span>' if unit == "%" and v is not None and top else ""
            cells.append(f'<td class="{"best" if v == best and v is not None else ""}">{_fmt(v, unit)}{bar}</td>')
        body.append(f"<tr><td>{html.escape(label)}</td>{''.join(cells)}</tr>")
    config_rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{html.escape(runs[n][1]['config'].get('retrieval', ''))}</td></tr>" for n in names)

    cmp_html = []
    for a, b in pairs:
        r = compare(runs[a][0], runs[b][0])
        pr = r["paired"]
        flips = "".join(
            f'<li>{html.escape(f["question"])}<br><span class="tag">{html.escape(a)}: {f["baseline"]}</span> '
            f'<span class="tag">{html.escape(b)}: {f["candidate"]}</span> <span class="muted">{html.escape(f["reason"] or "")}</span></li>'
            for f in r["flips"])
        fails = "".join(f"<li>{html.escape(x)}</li>" for x in r["gate"]["fails"])
        cmp_html.append(
            f'<div class="cmp"><h3>{html.escape(a)} → {html.escape(b)} {_pill(r["gate"]["verdict"])}</h3>'
            f'<p class="stat">accuracy {100 * pr["diff"]:+.1f} pts · 95% CI {100 * pr["ci95"][0]:+.1f} to {100 * pr["ci95"][1]:+.1f} · '
            f'only {html.escape(a)} right: {pr["only_baseline"]} · only {html.escape(b)} right: {pr["only_candidate"]} · McNemar p = {pr["p_value"]:.4f}</p>'
            + (f"<ul>{fails}</ul>" if fails else "")
            + (f"<details><summary>{len(r['flips'])} questions changed outcome</summary><ul class=\"flips\">{flips}</ul></details>" if r["flips"] else "")
            + "</div>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1>
<p class="muted">{html.escape(notes)}</p>
<div class="meta"><span><b>{first['questions']}</b> questions</span><span><b>{first['answerable']}</b> answerable</span>
<span><b>{first['out_of_scope']}</b> out of scope</span><span>model <b>{html.escape(first['config'].get('model', ''))}</b></span>
<span>judge <b>{html.escape(first['config'].get('judge', ''))}</b></span></div>
<h2>Runs</h2><div class="scroll"><table><thead><tr><th>Metric</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
<p class="muted" style="margin-top:8px;font-size:13px">Wrong answers shown = answers delivered to the user that the judge marked partial or incorrect, over all delivered answers. Over-escalation = answerable questions sent to a human.</p>
<h2>Configurations</h2><div class="scroll"><table><tbody>{config_rows}</tbody></table></div>
<h2>Should it ship?</h2>{''.join(cmp_html) or '<p class="muted">No comparisons requested.</p>'}
<footer>Generated by rag-eval-bench. Gate: block if accuracy drops with p &lt; 0.05, if wrong answers shown rise more than 2 points, or if out-of-scope escalation falls.</footer>
</main></body></html>"""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
