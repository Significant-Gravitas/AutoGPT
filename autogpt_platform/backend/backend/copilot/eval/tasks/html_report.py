"""A one-page HTML view of a task eval run: the score per reviewer, the
trend across saved runs, every category, and the three weakest.

    poetry run python -m backend.copilot.eval.tasks.html_report <run folder>

``__main__`` writes it next to ``report.md`` after every run. The page is
self-contained: data inlined, no network, motion only on transform and
opacity, and none at all under ``prefers-reduced-motion``.
"""

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
_COLORS = ["#E2875F", "#8EA2FF", "#5FD0A8"]
_LABELS = {
    "claude-opus-5-5": "Opus 5.5",
    "grok-4.7-medium-fast": "Grok 4.7",
    "gpt-5.4": "GPT-5.4",
}


def write_html(out: Path, summary: dict[str, Any]) -> Path:
    path = out / "report.html"
    path.write_text(render(page_data(out.name, summary)))
    return path


def page_data(run: str, summary: dict[str, Any]) -> dict[str, Any]:
    models = summary["models"]
    reviewers = models.get("reviewers") or [models.get("judge", "reviewer")]
    overall = _per_reviewer(summary["overall"], reviewers)
    categories = [
        {
            "name": name.replace("_", " ").capitalize().replace(" qa", " Q&A"),
            "scores": [s["score"] for s in _per_reviewer(stats, reviewers)],
        }
        for name, stats in summary["categories"].items()
    ]
    categories.sort(key=lambda c: -sum(c["scores"]))
    for c in categories:
        c["mean"] = round(sum(c["scores"]) / len(c["scores"]), 1)
    weakest = sorted(categories, key=lambda c: c["mean"])[:3]
    cases = overall[0]["cases"]
    return {
        "run": run,
        "subtitle": f"{cases} cases · 10 simulated companies · "
        f"Otto on {_label(models.get('otto', ''))}",
        "reviewers": [
            {
                "label": _label(name),
                "color": _COLORS[i % len(_COLORS)],
                "score": stats["score"],
                "pass": stats["pass_rate"],
            }
            for i, (name, stats) in enumerate(zip(reviewers, overall))
        ],
        "categories": categories,
        "weakest": [{"name": c["name"], "score": c["mean"]} for c in weakest],
        "history": history(),
    }


def history() -> list[dict[str, Any]]:
    """The saved run summaries next to this file, oldest first."""
    points = []
    paths = sorted(HERE.glob("baseline_*.json"), key=lambda p: p.stat().st_mtime)
    for index, path in enumerate(paths, start=1):
        data = json.loads(path.read_text())
        overall = data["overall"]
        score = (
            overall["score"]
            if "score" in overall
            else next(iter(overall.values()))["score"]
        )
        otto = _label(data["models"].get("otto", ""))
        points.append({"label": f"Run {index} · {otto}", "score": score})
    return points


def _per_reviewer(stats: dict[str, Any], reviewers: list[str]) -> list[dict[str, Any]]:
    if "score" in stats:  # a single-reviewer summary
        return [stats]
    return [stats[name] for name in reviewers]


def _label(name: str) -> str:
    short = name.split(":")[-1]
    return _LABELS.get(short, short)


def render(data: dict[str, Any], embed: bool = False) -> str:
    """The page. ``embed`` fits it inside a host that supplies the page
    background and theme colours (no own background, header or full height)."""
    page = _TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    return page.replace("/*EMBED*/", _EMBED_CSS) if embed else page


_EMBED_CSS = """
  :root { --x-bg: var(--background); --x-panel: var(--card); --x-line: var(--border);
          --x-text: var(--foreground); --x-muted: var(--muted-foreground);
          --x-faint: var(--muted-foreground); }
  html, body { background: none; }
  body { padding: 0; display: block; }
  main { width: 100%; }
  header { display: none; }
  .card { border-radius: var(--radius); padding: 16px; }
  .row:hover { background: var(--accent) !important; }
"""


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Otto · task eval</title>
<style>
  :root {
    --x-bg: #0B0B0D; --x-panel: #131316; --x-line: #222227; --x-text: #EDEDEF;
    --x-muted: #8B8D98; --x-faint: #5B5D66;
    --ease: cubic-bezier(0.32, 0.72, 0, 1);
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    background: var(--x-bg); color: var(--x-text);
    font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased; font-variant-numeric: tabular-nums;
    display: grid; place-items: start center; padding: 64px 20px;
  }
  html { background: var(--x-bg); }
  main { width: min(720px, 100%); display: grid; gap: 20px; }
  .rise { opacity: 0; transform: translateY(8px);
          transition: opacity 500ms var(--ease), transform 500ms var(--ease); }
  .rise.in { opacity: 1; transform: none; }
  header { display: grid; gap: 6px; padding: 0 4px 8px; }
  .eyebrow { color: var(--x-faint); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; }
  h1 { font-size: 28px; font-weight: 600; letter-spacing: -.02em; text-wrap: balance; }
  .sub { color: var(--x-muted); }
  .card { background: var(--x-panel); border: 1px solid var(--x-line); border-radius: 18px; padding: 24px; }
  .label { color: var(--x-muted); font-size: 13px; margin-bottom: 16px; }
  .gauges { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }
  .gauge { display: flex; align-items: center; gap: 18px; }
  .gauge svg { width: 104px; height: 104px; transform: rotate(-90deg); flex: none; }
  .gauge .track { stroke: var(--x-line); }
  .gauge .arc { transition: stroke-dashoffset 1400ms var(--ease); }
  .score { font-size: 40px; font-weight: 600; letter-spacing: -.03em; line-height: 1; }
  .score small { font-size: 15px; color: var(--x-faint); font-weight: 500; letter-spacing: 0; }
  .who { display: flex; align-items: center; gap: 8px; color: var(--x-muted); font-size: 13px; margin-top: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .trend svg { width: 100%; height: auto; overflow: visible; display: block; }
  .trend .path { fill: none; stroke: var(--x-text); stroke-width: 2; stroke-linecap: round;
                 stroke-linejoin: round; transition: stroke-dashoffset 1200ms var(--ease); }
  .trend .pt { opacity: 0; transform: scale(.6); transform-box: fill-box; transform-origin: center;
               transition: opacity 300ms var(--ease), transform 300ms var(--ease); }
  .trend .pt.in { opacity: 1; transform: none; }
  .trend text { fill: var(--x-muted); font-size: 11px; }
  .trend .val { fill: var(--x-text); font-size: 12px; font-weight: 600; }
  .rows { display: grid; gap: 4px; }
  .row { display: grid; grid-template-columns: 150px 1fr 44px; align-items: center; gap: 14px;
         padding: 8px 10px; margin: 0 -10px; border-radius: 10px; transition: background-color 150ms ease; }
  @media (hover: hover) and (pointer: fine) { .row:hover { background: #18181C; } }
  .name { color: var(--x-muted); font-size: 13px; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; }
  .bars { display: grid; gap: 4px; }
  .bar { height: 6px; border-radius: 3px; background: var(--x-line); overflow: hidden; }
  .fill { height: 100%; border-radius: 3px; transform: scaleX(0); transform-origin: left;
          transition: transform 900ms var(--ease); will-change: transform; }
  .num { text-align: right; font-size: 13px; font-weight: 600; }
  .weak { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
  .chip { border: 1px solid var(--x-line); border-radius: 14px; padding: 14px 16px; }
  .chip b { display: block; font-size: 22px; font-weight: 600; letter-spacing: -.02em; }
  .chip span { color: var(--x-muted); font-size: 13px; }
  footer { color: var(--x-faint); font-size: 12px; text-align: center; padding-top: 8px; }
  .legend { display: flex; gap: 16px; margin: -6px 0 14px; color: var(--x-muted); font-size: 12px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; animation: none !important; }
  }
/*EMBED*/
</style>
</head>
<body>
<main id="app"></main>
<script>
const D = __DATA__;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const ease = t => 1 - Math.pow(1 - t, 4);
const el = (tag, cls, html) => { const n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; };
const app = document.getElementById('app');

function countUp(node, to, ms) {
  if (reduce) { node.textContent = to.toFixed(1); return; }
  const start = performance.now();
  const tick = now => {
    const t = Math.min(1, (now - start) / ms);
    node.textContent = (to * ease(t)).toFixed(1);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

const head = el('header', 'rise');
head.append(el('div', 'eyebrow', 'Otto · task eval · ' + D.run),
            el('h1', null, 'How well Otto handles real work'),
            el('div', 'sub', D.subtitle));
app.append(head);

const gaugesCard = el('section', 'card rise');
gaugesCard.append(el('div', 'label', 'Score out of 100, one per reviewer'));
const gauges = el('div', 'gauges');
const R = 44, C = 2 * Math.PI * R;
const arcs = [];
D.reviewers.forEach(r => {
  const g = el('div', 'gauge');
  g.innerHTML = `<svg viewBox="0 0 104 104" aria-hidden="true">
      <circle class="track" cx="52" cy="52" r="${R}" fill="none" stroke-width="8"/>
      <circle class="arc" cx="52" cy="52" r="${R}" fill="none" stroke="${r.color}" stroke-width="8"
        stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C}"/></svg>
    <div><div class="score"><span>0.0</span><small> / 100</small></div>
    <div class="who"><span class="dot" style="background:${r.color}"></span>${r.label} · ${r.pass}% pass</div></div>`;
  arcs.push([g.querySelector('.arc'), g.querySelector('.score span'), r.score]);
  gauges.append(g);
});
gaugesCard.append(gauges);
app.append(gaugesCard);

if (D.history.length > 1) {
  const trend = el('section', 'card trend rise');
  trend.append(el('div', 'label', 'Score across runs'));
  const W = 640, H = 150, pad = 44, n = D.history.length;
  const all = D.history.map(p => p.score);
  const lo = Math.min(...all) - 4, hi = Math.max(...all) + 4;
  const x = i => pad + i * (W - 2 * pad) / (n - 1);
  const y = s => H - 22 - (s - lo) / (hi - lo) * (H - 44);
  const d = D.history.map((p, i) => (i ? 'L' : 'M') + x(i) + ' ' + y(p.score)).join(' ');
  trend.insertAdjacentHTML('beforeend', `<svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Scores across runs">
    <path class="path" d="${d}"/>
    ${D.history.map((p, i) => `<g class="pt" style="transition-delay:${300 + i * 180}ms">
      <circle cx="${x(i)}" cy="${y(p.score)}" r="4" fill="var(--x-bg)" stroke="var(--x-text)" stroke-width="2"/>
      <text class="val" x="${x(i)}" y="${y(p.score) - 12}" text-anchor="middle">${p.score}</text>
      <text x="${x(i)}" y="${H - 2}" text-anchor="middle">${p.label}</text></g>`).join('')}
  </svg>`);
  app.append(trend);
}

const cats = el('section', 'card rise');
cats.append(el('div', 'label', 'By category, average of both reviewers'));
cats.append(el('div', 'legend', D.reviewers.map(r =>
  `<span><i class="dot" style="background:${r.color}"></i>${r.label}</span>`).join('')));
const rows = el('div', 'rows');
const fills = [];
D.categories.forEach((c, i) => {
  const row = el('div', 'row');
  row.append(el('div', 'name', c.name));
  const bars = el('div', 'bars');
  c.scores.forEach((s, j) => {
    const bar = el('div', 'bar');
    const f = el('div', 'fill');
    f.style.background = D.reviewers[j].color;
    f.style.transitionDelay = (i * 45 + j * 60) + 'ms';
    bar.append(f); bars.append(bar); fills.push([f, s]);
  });
  row.append(bars, el('div', 'num', c.mean.toFixed(0)));
  rows.append(row);
});
cats.append(rows);
app.append(cats);

const weak = el('section', 'card rise');
weak.append(el('div', 'label', 'Weakest, worth fixing next'));
const chips = el('div', 'weak');
D.weakest.forEach(w => chips.append(el('div', 'chip', `<b>${w.score.toFixed(1)}</b><span>${w.name}</span>`)));
weak.append(chips);
app.append(weak);
app.append(el('footer', 'rise', 'Each case: half code checks, half the reviewer’s judgement.'));

const sections = [...document.querySelectorAll('.rise')];
function play(node) {
  node.classList.add('in');
  if (node === gaugesCard) arcs.forEach(([arc, num, s]) => {
    arc.style.strokeDashoffset = C * (1 - s / 100); countUp(num, s, 1400);
  });
  if (node === cats) fills.forEach(([f, s]) => f.style.transform = `scaleX(${s / 100})`);
  if (node.classList.contains('trend')) {
    const p = node.querySelector('.path'), len = p.getTotalLength();
    p.style.strokeDasharray = len; p.style.strokeDashoffset = reduce ? 0 : len;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      p.style.strokeDashoffset = 0;
      node.querySelectorAll('.pt').forEach(g => g.classList.add('in'));
    }));
  }
}
if (reduce) sections.forEach(play);
else {
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    if (e.isIntersecting) { io.unobserve(e.target); play(e.target); }
  }), { threshold: 0.2 });
  sections.forEach((s, i) => { s.style.transitionDelay = (i * 70) + 'ms'; io.observe(s); });
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    run = HERE / "runs" / sys.argv[1]
    print(write_html(run, json.loads((run / "summary.json").read_text())))
