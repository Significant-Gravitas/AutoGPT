"""Run the task eval and write a report.

    poetry run python -m backend.copilot.eval.tasks --limit 2 --concurrency 8

Needs OPENAI_API_KEY (Otto, the simulated user and the reviewer all run on
OpenAI models here). Writes ``runs/<timestamp>/`` next to this file:
``results.jsonl`` (every case, transcript and review), ``summary.json`` and
``report.md``.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import openai

from .judge import review, run_checks, score_case
from .models import Case, CaseResult, World
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


async def evaluate(args: argparse.Namespace) -> Path:
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=3)
    models = Models(client, otto=args.model, user=args.user_model)
    worlds = load_worlds()
    cases = [c for c in load_cases(args.category, args.limit) if c.world in worlds]
    out = HERE / "runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    gate = asyncio.Semaphore(args.concurrency)
    results: list[CaseResult] = []

    async def one(case: Case) -> None:
        async with gate:
            run = await run_case(case, worlds[case.world], models)
            checks = run_checks(case, run)
            verdict = await review(case, run, checks, client, args.judge_model)
            result = score_case(case, run, checks, verdict)
            results.append(result)
            with (out / "results.jsonl").open("a") as f:
                f.write(result.model_dump_json() + "\n")
            print(
                f"[{len(results)}/{len(cases)}] {case.id}: {result.verdict} {result.score}"
            )

    await asyncio.gather(*(one(c) for c in cases))
    write_report(out, results, args)
    return out


def write_report(
    out: Path, results: list[CaseResult], args: argparse.Namespace
) -> None:
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    def stats(rs: list[CaseResult]) -> dict[str, float]:
        n = len(rs) or 1
        return {
            "cases": len(rs),
            "score": round(sum(r.score for r in rs) / n, 1),
            "pass_rate": round(100 * sum(r.verdict == "pass" for r in rs) / n, 1),
            "partial": sum(r.verdict == "partial" for r in rs),
            "fail": sum(r.verdict == "fail" for r in rs),
        }

    runs = [r.run for r in results]
    summary = {
        "models": {
            "otto": args.model,
            "judge": args.judge_model,
            "user": args.user_model,
        },
        "overall": stats(results),
        "categories": {k: stats(v) for k, v in sorted(by_category.items())},
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
    lines = [
        f"# Task eval — {out.name}",
        "",
        f"Otto: `{args.model}` · reviewer: `{args.judge_model}` · user: `{args.user_model}`",
        "",
        f"**Overall: {summary['overall']['score']} / 100**, pass rate "
        f"{summary['overall']['pass_rate']}% over {summary['overall']['cases']} cases.",
        "",
        "| Category | Cases | Score | Pass % | Partial | Fail |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {v['cases']} | {v['score']} | {v['pass_rate']} | {v['partial']} | {v['fail']} |"
        for k, v in summary["categories"].items()
    ]
    lines += ["", "## Cases that did not pass", ""]
    for r in sorted(results, key=lambda r: r.score):
        if r.verdict == "pass":
            continue
        failed = ", ".join(c.name for c in r.checks if not c.passed) or "none"
        lines.append(f"- **{r.case_id}** ({r.verdict}, {r.score}) {r.title}")
        lines.append(f"  - failed checks: {failed}")
        lines += [f"  - {reason}" for reason in r.review.reasons[:3]]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", action="append")
    parser.add_argument("--limit", type=int, help="cases per category file")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--judge-model", default="gpt-5.4")
    parser.add_argument("--user-model", default="gpt-5.4-mini")
    out = asyncio.run(evaluate(parser.parse_args()))
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
