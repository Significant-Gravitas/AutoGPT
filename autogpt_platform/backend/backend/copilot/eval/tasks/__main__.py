"""Run the task eval and write a report.

    poetry run python -m backend.copilot.eval.tasks \\
        --backend claude --model claude-opus-5-5 --user-model claude-sonnet-5-5 \\
        --reviewer claude:claude-opus-5-5 --reviewer cursor:grok-4.7-medium-fast

Otto and the simulated user run on ``--backend``: the OpenAI API (needs
OPENAI_API_KEY), ``claude -p`` on a Claude subscription, or ``cursor-agent``
on a Cursor subscription. Every ``--reviewer`` rates every case, so the
report carries one score per reviewer. Writes ``runs/<timestamp>/`` next to
this file: ``results.jsonl`` (every case, transcript and ratings),
``summary.json`` and ``report.md``.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import openai

from .html_report import write_html
from .judge import review, run_checks, score_case
from .llm import Backend, ClaudeBackend, CursorBackend, OpenAIBackend
from .models import Case, CaseResult, Rating, World
from .runner import Models, run_case

HERE = Path(__file__).parent
DATA = HERE / "data"


def load_worlds() -> dict[str, World]:
    return {
        p.stem: World.model_validate_json(p.read_text())
        for p in sorted((DATA / "worlds").glob("*.json"))
    }


def load_cases(categories: list[str] | None, limit: int | None) -> list[Case]:
    cases: list[Case] = []
    for path in sorted((DATA / "cases").glob("*.json")):
        batch = [Case.model_validate(c) for c in json.loads(path.read_text())]
        batch = [c for c in batch if not categories or c.category in categories]
        cases += batch[:limit] if limit else batch
    return cases


def make_backend(name: str) -> Backend:
    if name == "cursor":
        return CursorBackend()
    if name == "claude":
        return ClaudeBackend()
    return OpenAIBackend(
        openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3)
    )


def parse_reviewers(specs: list[str]) -> list[tuple[str, Backend, str]]:
    """``backend:model`` per reviewer, e.g. ``claude:claude-opus-5-5``."""
    reviewers = []
    for spec in specs:
        backend, _, model = spec.partition(":")
        reviewers.append((spec, make_backend(backend), model))
    return reviewers


def new_run_dir() -> Path:
    out = HERE / "runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    return out


def resume(name: str) -> tuple[list[CaseResult], Path]:
    """Keep a run's cases that finished cleanly, and rerun the rest into it."""
    out = HERE / "runs" / name
    path = out / "results.jsonl"
    results = [
        CaseResult.model_validate_json(line) for line in path.read_text().splitlines()
    ]
    kept = [r for r in results if not r.run.error and not _review_failed(r)]
    path.write_text("".join(r.model_dump_json() + "\n" for r in kept))
    return kept, out


def _review_failed(result: CaseResult) -> bool:
    return any(
        reason.startswith("review failed")
        for rating in result.ratings.values()
        for reason in rating.review.reasons
    )


async def evaluate(args: argparse.Namespace) -> Path:
    models = Models(make_backend(args.backend), otto=args.model, user=args.user_model)
    reviewers = parse_reviewers(args.reviewer or [f"{args.backend}:{args.model}"])
    worlds = load_worlds()
    cases = [c for c in load_cases(args.category, args.limit) if c.world in worlds]
    results, out = resume(args.resume) if args.resume else ([], new_run_dir())
    done = {r.case_id for r in results}
    cases = [c for c in cases if c.id not in done]
    gate = asyncio.Semaphore(args.concurrency)

    async def one(case: Case) -> None:
        async with gate:
            run = await run_case(case, worlds[case.world], models)
        checks = run_checks(case, run)
        reviews = await asyncio.gather(
            *(review(case, run, checks, b, m) for _, b, m in reviewers)
        )
        scored = [score_case(case, run, checks, r) for r in reviews]
        result = scored[0].model_copy(
            update={
                "ratings": {
                    name: Rating(review=s.review, score=s.score, verdict=s.verdict)
                    for (name, _, _), s in zip(reviewers, scored)
                }
            }
        )
        results.append(result)
        with (out / "results.jsonl").open("a") as f:
            f.write(result.model_dump_json() + "\n")
        marks = " | ".join(
            f"{name.split(':')[-1]}: {r.verdict} {r.score}"
            for name, r in result.ratings.items()
        )
        print(f"[{len(results)}] {case.id}: {marks}", flush=True)

    await asyncio.gather(*(one(c) for c in cases))
    write_report(out, results, args, [name for name, _, _ in reviewers])
    return out


def write_report(
    out: Path,
    results: list[CaseResult],
    args: argparse.Namespace,
    reviewers: list[str],
) -> None:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    def stats(rs: list[CaseResult], name: str) -> dict[str, float]:
        n = len(rs) or 1
        ratings = [r.ratings[name] for r in rs]
        return {
            "cases": len(rs),
            "score": round(sum(x.score for x in ratings) / n, 1),
            "pass_rate": round(100 * sum(x.verdict == "pass" for x in ratings) / n, 1),
            "partial": sum(x.verdict == "partial" for x in ratings),
            "fail": sum(x.verdict == "fail" for x in ratings),
        }

    runs = [r.run for r in results]
    summary = {
        "models": {
            "backend": args.backend,
            "otto": args.model,
            "user": args.user_model,
            "reviewers": reviewers,
        },
        "overall": {name: stats(results, name) for name in reviewers},
        "categories": {
            k: {name: stats(v, name) for name in reviewers}
            for k, v in sorted(by_category.items())
        },
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in runs),
            "completion": sum(r.completion_tokens for r in runs),
        },
        "seconds_per_turn": round(
            sum(t.seconds for r in runs for t in r.turns)
            / max(1, sum(len(r.turns) for r in runs)),
            1,
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_html(out, summary)
    short = [name.split(":")[-1] for name in reviewers]
    lines = [
        f"# Task eval — {out.name}",
        "",
        f"Otto: `{args.backend}:{args.model}` · simulated user: `{args.user_model}`",
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
        "| Category | Cases | "
        + " | ".join(f"{s} score | {s} pass %" for s in short)
        + " |",
        "|---|---|" + "---|---|" * len(short),
    ]
    for k, per in summary["categories"].items():
        cells = " | ".join(
            f"{per[n]['score']} | {per[n]['pass_rate']}" for n in reviewers
        )
        lines.append(f"| {k} | {per[reviewers[0]]['cases']} | {cells} |")
    lines += ["", "## Cases not passed by every reviewer", ""]
    for r in sorted(results, key=lambda r: min(x.score for x in r.ratings.values())):
        if all(x.verdict == "pass" for x in r.ratings.values()):
            continue
        failed = ", ".join(c.name for c in r.checks if not c.passed) or "none"
        marks = ", ".join(
            f"{n.split(':')[-1]} {x.verdict} {x.score}" for n, x in r.ratings.items()
        )
        lines.append(f"- **{r.case_id}** ({marks}) {r.title}")
        lines.append(f"  - failed checks: {failed}")
        for n, x in r.ratings.items():
            lines += [
                f"  - {n.split(':')[-1]}: {reason}" for reason in x.review.reasons[:2]
            ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append")
    parser.add_argument(
        "--resume",
        help="a run folder name under runs/: keep its clean cases, rerun the rest",
    )
    parser.add_argument("--limit", type=int, help="cases per category file")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--backend",
        choices=["openai", "cursor", "claude"],
        default="openai",
        help="Where Otto and the simulated user run. openai: the API, needs "
        "OPENAI_API_KEY. cursor: cursor-agent on a Cursor subscription. "
        "claude: claude -p on a Claude subscription.",
    )
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument(
        "--reviewer",
        action="append",
        help="backend:model, repeatable; each reviewer rates every case, e.g. "
        "--reviewer claude:claude-opus-5-5 --reviewer cursor:grok-4.7-medium-fast",
    )
    parser.add_argument("--user-model", default="gpt-5.4-mini")
    out = asyncio.run(evaluate(parser.parse_args()))
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
