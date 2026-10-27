"""User-pronoun eval (SECRT-3161): does Otto, or an expert, guess the gender
of a user who never stated it?

    poetry run python -m backend.copilot.eval.pronouns.runner [--repeats 3]

Runs every case in ``cases.json`` through the production system prompt, tools
and routed model, then checks the text in which the user is the only third
person. Like the style eval it is built on (``generate_turn``), every turn goes
over the baseline engine's OpenAI-compatible transport, even when production
routes the model's mode (``thinking``) to the SDK engine; the run says so.
Checked text:

- ``otto_brief``: what Otto writes into each delegation or handoff call. A
  turn that delegates nothing is ``no_delegation``: inconclusive, not a pass.
- ``expert_task``: the expert's report back to Otto.

Any he/she/him/her/his/hers there fails the case. The run exits 1 on any
failed or errored case, and unless every persona's every brief and task has at
least one passing result: an owner whose cases never ran, or whose Otto briefs
never delegated, has not been checked at all.
"""

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.eval.style.assembly import resolve_chat_model
from backend.copilot.eval.style.generation import Turn, chat_client, generate_turn
from backend.copilot.model_normalize import normalize_model_for_transport

from .cases import (
    CaseKind,
    PronounCase,
    expand_cases,
    first_message,
    gendered_pronouns,
    load_case_file,
    teammate_expert,
)
from .delegation import delegated_tasks

Outcome = Literal["pass", "fail", "no_delegation", "error"]


class CaseResult(BaseModel):
    case_id: str
    kind: CaseKind
    repeat: int
    outcome: Outcome
    pronouns: list[str] = []
    checked: list[str] = []
    error: str | None = None


def score(case: PronounCase, turn: Turn, repeat: int = 1) -> CaseResult:
    """Pass, fail or inconclusive for one generated turn."""
    checked = delegated_tasks(turn) if case.kind == "otto_brief" else [turn.text]
    checked = [text for text in checked if text.strip()]
    if not checked:
        outcome: Outcome = "no_delegation" if case.kind == "otto_brief" else "error"
        return CaseResult(
            case_id=case.id, kind=case.kind, repeat=repeat, outcome=outcome
        )
    pronouns = [p for text in checked for p in gendered_pronouns(text)]
    incomplete = _incomplete(case, turn)
    return CaseResult(
        case_id=case.id,
        kind=case.kind,
        repeat=repeat,
        outcome="fail" if pronouns else "error" if incomplete else "pass",
        pronouns=pronouns,
        checked=checked,
        error=None if pronouns else incomplete,
    )


def _incomplete(case: PronounCase, turn: Turn) -> str | None:
    """Why the checked text may be partial, so a clean read can't pass it.

    A pronoun already in partial text still fails the case. A truncated turn
    may have cut a call's arguments short; an expert report stopped at the
    round cap was never finished. Otto's delegations at the round cap were
    whole calls when sent, so they still score.
    """
    if turn.truncated:
        return "turn hit the token limit"
    if case.kind == "expert_task" and turn.hit_round_cap:
        return "report stopped at the tool round cap"
    return None


async def run(*, repeats: int, model: str | None, concurrency: int) -> list[CaseResult]:
    config = ChatConfig()
    case_file = load_case_file()
    teammate = teammate_expert(case_file.teammate)
    routed = await resolve_chat_model(config)
    route = model or routed.slug
    chat_model = normalize_model_for_transport(route, config)
    client = chat_client(config)
    semaphore = asyncio.Semaphore(concurrency)
    print(
        f"model: {route}, routed mode: {routed.mode}, sent over the baseline "
        "transport (production runs the thinking mode on the SDK engine)",
        file=sys.stderr,
    )

    async def one(case: PronounCase, repeat: int) -> CaseResult:
        async with semaphore:
            try:
                turn = await generate_turn(
                    client,
                    config,
                    model=chat_model,
                    expert=None if case.kind == "otto_brief" else teammate,
                    roster=[teammate],
                    user_message=first_message(case, teammate),
                )
            except Exception as e:
                return CaseResult(
                    case_id=case.id,
                    kind=case.kind,
                    repeat=repeat,
                    outcome="error",
                    error=str(e),
                )
            return score(case, turn, repeat)

    jobs = [
        one(case, repeat)
        for case in expand_cases(case_file)
        for repeat in range(1, repeats + 1)
    ]
    return list(await asyncio.gather(*jobs))


def summarize(results: list[CaseResult]) -> str:
    counts = Counter(r.outcome for r in results)
    lines = [
        f"{r.case_id} #{r.repeat}: {r.outcome}"
        + (f" ({', '.join(r.pronouns)})" if r.pronouns else "")
        + (f" [{r.error}]" if r.error else "")
        for r in results
    ]
    totals = ", ".join(f"{counts[o]} {o}" for o in get_args(Outcome))
    unscored = unscored_cases(results)
    if unscored:
        lines.append(f"not scored ({len(unscored)}): {', '.join(unscored)}")
    return "\n".join([*lines, f"{len(results)} runs: {totals}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--model", help="override the routed chat model")
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--out", type=Path, help="write every result as JSON")
    args = parser.parse_args()
    results = asyncio.run(
        run(repeats=args.repeats, model=args.model, concurrency=args.concurrency)
    )
    if args.out:
        args.out.write_text(
            "[\n" + ",\n".join(r.model_dump_json(indent=2) for r in results) + "\n]\n",
            encoding="utf-8",
        )
    print(summarize(results))
    sys.exit(exit_code(results))


def exit_code(results: list[CaseResult], cases: list[PronounCase] | None = None) -> int:
    """0 only when nothing failed or errored and every case was scored clean.

    ``cases`` defaults to every persona against every brief and task in
    ``cases.json``. A case with no passing result (never run, or only ever
    ``no_delegation``) leaves the run incomplete: an owner the eval never
    exercised has proved nothing about how they are referred to.
    """
    if any(r.outcome in ("fail", "error") for r in results):
        return 1
    return 1 if unscored_cases(results, cases) else 0


def unscored_cases(
    results: list[CaseResult], cases: list[PronounCase] | None = None
) -> list[str]:
    """The ids of expected cases with no passing result, in case order."""
    expected = cases if cases is not None else expand_cases(load_case_file())
    passed = {r.case_id for r in results if r.outcome == "pass"}
    return [case.id for case in expected if case.id not in passed]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


if __name__ == "__main__":
    main()
