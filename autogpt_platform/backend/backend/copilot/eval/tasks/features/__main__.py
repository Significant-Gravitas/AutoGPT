"""Run the task features eval (task controls, suggested next tasks) and
write a report.

    # list the cases and what a real run would cost; no model call
    poetry run python -m backend.copilot.eval.tasks.features --dry-run

    # code-only cases: no model, no database, free
    poetry run python -m backend.copilot.eval.tasks.features --offline

    # everything but the live tier
    poetry run python -m backend.copilot.eval.tasks.features \\
        --backend claude-native --model claude-opus-5-5 \\
        --drafter claude:claude-haiku-5-5 \\
        --reviewer claude:claude-opus-5-5 --reviewer cursor:grok-4.7-medium-fast

    # add the live tier: needs Postgres, Redis, RabbitMQ and a copilot executor
    ... --live --live-user-id <an existing dev user id>

Writes ``runs/<timestamp>/`` next to this file: ``results.jsonl``,
``summary.json``, ``report.md`` and ``report.html``.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from backend.copilot.eval.tasks.__main__ import (
    load_worlds,
    make_backend,
    parse_reviewers,
)
from backend.copilot.eval.tasks.native.runner import run_case_native
from backend.copilot.eval.tasks.runner import Models, run_case
from backend.data import db

from .cost import case_usage, dollars
from .live import RealTaskApi
from .models import FeatureCase, FeatureResult
from .play import Settings, play
from .report import write_report
from .suggestion_run import Drafter

HERE = Path(__file__).parent
CASES = HERE / "data" / "cases"


def load_cases(
    areas: list[str] | None = None,
    ids: list[str] | None = None,
    needs: list[str] | None = None,
) -> list[FeatureCase]:
    cases = [
        FeatureCase.model_validate(c)
        for path in sorted(CASES.glob("*.json"))
        for c in json.loads(path.read_text())
    ]
    return [
        c
        for c in cases
        if (not areas or c.area in areas)
        and (not ids or c.id in ids)
        and (not needs or c.needs in needs)
    ]


def dry_run(cases: list[FeatureCase], args: argparse.Namespace) -> str:
    reviewers = args.reviewer or [f"{args.backend}:{args.model}"]
    drafter = args.drafter.split(":")[-1]
    lines = [f"{'case':<20} {'area':<12} {'needs':<6} review  est $   title"]
    total, unpriced = 0.0, set()
    for case in cases:
        usage = case_usage(case, args.model, drafter, reviewers, args.live_model)
        cost, missing = dollars(usage)
        total += cost
        unpriced |= set(missing)
        flag = "yes" if case.review else "-"
        lines.append(
            f"{case.id:<20} {case.area:<12} {case.needs:<6} {flag:<7} "
            f"{cost:>6.3f}  {case.title}"
        )
    counts = {n: sum(c.needs == n for c in cases) for n in ("none", "model", "live")}
    lines += [
        "",
        f"{len(cases)} cases: {counts['none']} code-only, {counts['model']} "
        f"model, {counts['live']} live; {sum(c.review for c in cases)} reviewed",
        f"Estimated API-equivalent cost: ${total:.2f} "
        f"(worker {args.backend}:{args.model}, drafter {args.drafter}, "
        f"reviewers {', '.join(reviewers)}, live worker {args.live_model})",
    ]
    if unpriced:
        lines.append(
            f"Not priced (no catalog rate or a subscription): {', '.join(sorted(unpriced))}"
        )
    return "\n".join(lines)


def settings_for(args: argparse.Namespace) -> Settings:
    worlds = load_worlds()
    if args.offline:
        return Settings(worlds)
    backend = make_backend(args.backend)
    drafter_backend, _, drafter_model = args.drafter.partition(":")
    return Settings(
        worlds,
        models=Models(backend, otto=args.model, user=args.user_model),
        engine=run_case_native if args.backend == "claude-native" else run_case,
        reviewers=parse_reviewers(args.reviewer or [f"{args.backend}:{args.model}"]),
        drafter=Drafter(make_backend(drafter_backend), drafter_model),
        live_api=RealTaskApi() if args.live else None,
        live_user_id=args.live_user_id,
    )


async def evaluate(args: argparse.Namespace) -> Path:
    cases = load_cases(args.area, args.case, args.needs)
    settings = settings_for(args)
    out = HERE / "runs" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    gate = asyncio.Semaphore(args.concurrency)
    results: list[FeatureResult] = []
    if args.live:
        await db.connect()

    async def one(case: FeatureCase) -> None:
        async with gate:
            result = await play(case, settings)
        results.append(result)
        with (out / "results.jsonl").open("a") as f:
            f.write(result.model_dump_json() + "\n")
        mark = (
            result.skipped
            and "skipped"
            or " | ".join(
                f"{n.split(':')[-1]}: {r.verdict} {r.score}"
                for n, r in result.ratings.items()
            )
        )
        print(f"[{len(results)}] {case.id}: {mark}", flush=True)

    try:
        await asyncio.gather(*(one(c) for c in cases))
    finally:
        if args.live:
            await db.disconnect()
    setup = {
        "backend": args.backend,
        "model": None if args.offline else args.model,
        "drafter": None if args.offline else args.drafter,
        "tiers": sorted({c.needs for c in cases}),
        "offline": args.offline,
        "live": args.live,
    }
    write_report(out, results, setup, settings.names)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--area", action="append", choices=["controls", "suggestions"])
    parser.add_argument("--case", action="append", help="run only these case ids")
    parser.add_argument(
        "--needs",
        action="append",
        choices=["none", "model", "live"],
        help="run only cases needing this",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list cases and estimate cost; no calls"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="no model calls: code checks only, model cases skipped, no reviews",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="run live cases against the database and services",
    )
    parser.add_argument(
        "--live-user-id", help="an existing user the live cases make their tasks for"
    )
    parser.add_argument(
        "--live-model",
        default="claude-opus-5-5",
        help="for the cost estimate: the model the platform's worker runs on",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument(
        "--backend",
        choices=["openai", "cursor", "claude", "claude-native"],
        default="claude-native",
        help="where the worker turns run, as in the task eval",
    )
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--user-model", default="claude-sonnet-5-5")
    parser.add_argument(
        "--drafter",
        default="claude:claude-haiku-5-5",
        help="backend:model for drafting suggestions; production uses the aux tier (Haiku 5.5)",
    )
    parser.add_argument(
        "--reviewer",
        action="append",
        help="backend:model, repeatable, as in the task eval",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(dry_run(load_cases(args.area, args.case, args.needs), args))
        return
    if args.live and not args.live_user_id:
        parser.error("--live needs --live-user-id")
    out = asyncio.run(evaluate(args))
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
