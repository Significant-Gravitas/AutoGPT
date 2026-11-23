"""Write a feature eval run's ``summary.json``, ``report.md`` and
``report.html``, in the task eval's format: a score per reviewer, a row per
area, and every case not passed with its failed checks and the reviewers'
reasons. Skipped cases are listed with why, and count nowhere."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.copilot.eval.tasks.html_report import render

from .models import FeatureResult

HERE = Path(__file__).parent
_COLORS = ["#E2875F", "#8EA2FF", "#5FD0A8"]


def write_report(
    out: Path, results: list[FeatureResult], setup: dict[str, Any], reviewers: list[str]
) -> dict[str, Any]:
    ran = [r for r in results if not r.skipped]
    by_area: dict[str, list[FeatureResult]] = defaultdict(list)
    for r in ran:
        by_area[r.area].append(r)
    summary = {
        "setup": setup,
        "reviewers": reviewers,
        "overall": {name: _stats(ran, name) for name in reviewers},
        "areas": {
            area: {name: _stats(rs, name) for name in reviewers}
            for area, rs in sorted(by_area.items())
        },
        "by_needs": {
            needs: sum(1 for r in ran if r.needs == needs)
            for needs in ("none", "model", "live")
        },
        "skipped": {r.case_id: r.skipped for r in results if r.skipped},
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in ran),
            "completion": sum(r.completion_tokens for r in ran),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(_markdown(out.name, summary, ran, reviewers))
    (out / "report.html").write_text(render(_page(out.name, summary, reviewers)))
    return summary


def _stats(rs: list[FeatureResult], name: str) -> dict[str, float]:
    n = len(rs) or 1
    ratings = [r.ratings[name] for r in rs if name in r.ratings]
    return {
        "cases": len(rs),
        "score": round(sum(x.score for x in ratings) / n, 1),
        "pass_rate": round(100 * sum(x.verdict == "pass" for x in ratings) / n, 1),
        "partial": sum(x.verdict == "partial" for x in ratings),
        "fail": sum(x.verdict == "fail" for x in ratings),
    }


def _markdown(
    run: str, summary: dict[str, Any], ran: list[FeatureResult], reviewers: list[str]
) -> str:
    setup = summary["setup"]
    short = [name.split(":")[-1] for name in reviewers]
    lines = [
        f"# Task features eval — {run}",
        "",
        f"Worker: `{setup.get('backend')}:{setup.get('model')}` · drafter: "
        f"`{setup.get('drafter')}` · tiers run: {', '.join(setup.get('tiers', []))}",
        "",
    ]
    lines += [
        f"- **{label}: {summary['overall'][name]['score']} / 100**, pass rate "
        f"{summary['overall'][name]['pass_rate']}% over "
        f"{summary['overall'][name]['cases']} cases"
        for name, label in zip(reviewers, short)
    ]
    lines += [
        "",
        "| Area | Cases | "
        + " | ".join(f"{s} score | {s} pass %" for s in short)
        + " |",
        "|---|---|" + "---|---|" * len(short),
    ]
    for area, per in summary["areas"].items():
        cells = " | ".join(
            f"{per[n]['score']} | {per[n]['pass_rate']}" for n in reviewers
        )
        lines.append(f"| {area} | {per[reviewers[0]]['cases']} | {cells} |")
    lines += ["", "## Cases not passed", ""]
    for r in sorted(
        ran, key=lambda r: min((x.score for x in r.ratings.values()), default=0)
    ):
        if r.ratings and all(x.verdict == "pass" for x in r.ratings.values()):
            continue
        marks = ", ".join(
            f"{n.split(':')[-1]} {x.verdict} {x.score}" for n, x in r.ratings.items()
        )
        lines.append(f"- **{r.case_id}** ({marks}) {r.title}")
        for c in r.checks:
            if not c.passed:
                lines.append(f"  - failed check `{c.name}`: {c.detail}")
        for n, x in r.ratings.items():
            lines += [f"  - {n.split(':')[-1]}: {why}" for why in x.review.reasons[:2]]
    if summary["skipped"]:
        lines += ["", "## Skipped", ""]
        lines += [f"- **{cid}**: {why}" for cid, why in summary["skipped"].items()]
    return "\n".join(lines) + "\n"


def _page(run: str, summary: dict[str, Any], reviewers: list[str]) -> dict[str, Any]:
    overall = [summary["overall"][name] for name in reviewers]
    areas = [
        {"name": area.capitalize(), "scores": [per[n]["score"] for n in reviewers]}
        for area, per in summary["areas"].items()
    ]
    for a in areas:
        a["mean"] = round(sum(a["scores"]) / len(a["scores"]), 1)
    cases = overall[0]["cases"] if overall else 0
    return {
        "run": run,
        "subtitle": f"{cases} cases · task controls, suggested tasks",
        "reviewers": [
            {
                "label": name.split(":")[-1],
                "color": _COLORS[i % len(_COLORS)],
                "score": stats["score"],
                "pass": stats["pass_rate"],
            }
            for i, (name, stats) in enumerate(zip(reviewers, overall))
        ],
        "categories": sorted(areas, key=lambda a: -a["mean"]),
        "weakest": [
            {"name": a["name"], "score": a["mean"]}
            for a in sorted(areas, key=lambda a: a["mean"])[:3]
        ],
        "history": [],
    }
