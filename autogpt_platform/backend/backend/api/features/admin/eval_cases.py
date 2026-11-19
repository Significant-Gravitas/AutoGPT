"""Turn the cases a run saved (``results.jsonl``) into the page's cases, with
what each case expected from its definition in the eval's ``data/cases``."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from . import eval_files
from .eval_files import load_json_list, parse_or_none, read_result_lines
from .eval_models import (
    EvalCase,
    EvalCheck,
    EvalExpectation,
    EvalQuestion,
    EvalRating,
    EvalSuite,
    EvalToolCall,
    EvalTurn,
    EvalVerdict,
)

ARGUMENTS_PREVIEW = 2000
OUTPUT_PREVIEW = 20000
_VERDICTS: dict[str, EvalVerdict] = {
    "pass": "pass",
    "partial": "partial",
    "fail": "fail",
}


class _Raw(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _RawReview(_Raw):
    verdict: str = "fail"
    score: float = 0
    reasons: list[str] = Field(default_factory=list)


class _RawRating(_Raw):
    review: _RawReview = Field(default_factory=_RawReview)
    score: float = 0
    verdict: str = "fail"


class _RawToolCall(_Raw):
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    service: str | None = None
    refused: str | None = None
    held: bool = False
    wrote: bool = False
    failed: bool = False
    result_preview: str = ""


class _RawTurn(_Raw):
    user: str = ""
    reply: str = ""
    final_reply: str = ""
    tool_calls: list[_RawToolCall] = Field(default_factory=list)
    questions: list[EvalQuestion] = Field(default_factory=list)
    ended_by: str = "reply"
    seconds: float = 0


class _RawRun(_Raw):
    turns: list[_RawTurn] = Field(default_factory=list)
    error: str | None = None


class _RawResult(_Raw):
    case_id: str
    category: str | None = None
    area: str | None = None
    title: str = ""
    world: str = ""
    run: _RawRun | None = None
    checks: list[EvalCheck] = Field(default_factory=list)
    review: _RawReview | None = None
    score: float | None = None
    verdict: str | None = None
    ratings: dict[str, _RawRating] = Field(default_factory=dict)
    skipped: str | None = None
    error: str | None = None
    needs: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)


class _RawTurnDef(_Raw):
    user: str = ""
    worker: dict[str, Any] | None = None
    task_result: dict[str, Any] | None = None


class _RawExpected(_Raw):
    behaviour: str | None = None
    must: list[str] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)


class _RawHuman(_Raw):
    reply: str = ""
    actions: list[str] = Field(default_factory=list)
    why: str = ""


class _RawTaskSpec(_Raw):
    title: str = ""
    description: str | None = None


class CaseDef(_Raw):
    """A case as the eval defines it: what the user asks and what is expected."""

    id: str
    category: str | None = None
    area: str | None = None
    title: str = ""
    turns: list[_RawTurnDef] = Field(default_factory=list)
    expected: _RawExpected = Field(default_factory=_RawExpected)
    human_example: _RawHuman = Field(default_factory=_RawHuman)
    task: _RawTaskSpec | None = None
    note: str | None = None


def load_case_defs(suite: EvalSuite) -> dict[str, CaseDef]:
    folder = eval_files.EVAL_DIR / "data" / "cases"
    if suite == "features":
        folder = eval_files.EVAL_DIR / "features" / "data" / "cases"
    defs = [CaseDef.model_validate(entry) for entry in load_json_list(folder)]
    return {d.id: d for d in defs}


def read_cases(
    path: Path, reviewers: list[str], defs: dict[str, CaseDef]
) -> list[EvalCase]:
    results = [parse_or_none(_RawResult, line) for line in read_result_lines(path)]
    return [to_case(r, reviewers, defs) for r in results if r is not None]


def to_case(
    raw: _RawResult, reviewers: list[str], defs: dict[str, CaseDef]
) -> EvalCase:
    ratings = _ratings(raw, reviewers)
    primary = ratings[0] if ratings and not raw.skipped else None
    definition = defs.get(raw.case_id)
    turns = [_turn(t) for t in raw.run.turns] if raw.run else []
    return EvalCase(
        case_id=raw.case_id,
        category=raw.category or raw.area or category_of(raw.case_id),
        title=raw.title or (definition.title if definition else ""),
        world=raw.world,
        verdict=primary.verdict if primary else None,
        score=primary.score if primary else None,
        request=request_text(definition) or (turns[0].user if turns else ""),
        expected=expectation(definition),
        observed=_observed(raw.run),
        ratings=ratings,
        checks=raw.checks,
        turns=turns,
        error=raw.error or (raw.run.error if raw.run else None),
        skipped=raw.skipped,
        needs=raw.needs,
        output=_output(raw.output),
    )


def verdict(value: str | None) -> EvalVerdict:
    return _VERDICTS.get(value or "", "fail")


def category_of(case_id: str) -> str:
    return case_id.rsplit("_", 1)[0]


def request_text(definition: CaseDef | None) -> str:
    """What starts the case, in a line a person can read."""
    if definition is None:
        return ""
    if definition.task is not None:
        parts = [definition.task.title, definition.task.description, definition.note]
        return "\n\n".join(p for p in parts if p)
    if not definition.turns:
        return definition.title
    first = definition.turns[0]
    if first.worker is not None:
        return f"Worker brief: {first.worker.get('title', '')}\n\n{first.worker.get('brief', '')}"
    if first.task_result is not None:
        title = first.task_result.get("title", "")
        return f"A task came back ({first.task_result.get('outcome', 'completed')}): {title}"
    return "\n\n".join(t.user for t in definition.turns if t.user)


def expectation(definition: CaseDef | None) -> EvalExpectation | None:
    if definition is None:
        return None
    human = definition.human_example
    return EvalExpectation(
        behaviour=definition.expected.behaviour,
        must=definition.expected.must,
        must_not=definition.expected.must_not,
        human_reply=human.reply,
        human_actions=human.actions,
        human_why=human.why,
    )


def _ratings(raw: _RawResult, reviewers: list[str]) -> list[EvalRating]:
    if not raw.ratings and raw.review is not None:
        name = reviewers[0] if reviewers else "reviewer"
        only = _RawRating(
            review=raw.review, score=raw.score or 0, verdict=raw.verdict or "fail"
        )
        return [_rating(name, only)]
    order = [r for r in reviewers if r in raw.ratings]
    order += [r for r in raw.ratings if r not in order]
    return [_rating(name, raw.ratings[name]) for name in order]


def _rating(name: str, raw: _RawRating) -> EvalRating:
    return EvalRating(
        reviewer=name,
        label=name.split(":")[-1],
        verdict=verdict(raw.verdict),
        score=raw.score,
        reasons=raw.review.reasons,
    )


def _turn(raw: _RawTurn) -> EvalTurn:
    return EvalTurn(
        user=raw.user,
        reply=raw.reply,
        final_reply=raw.final_reply,
        ended_by=raw.ended_by,
        seconds=raw.seconds,
        questions=raw.questions,
        tool_calls=[
            EvalToolCall(
                name=c.name,
                service=c.service,
                arguments=json.dumps(c.arguments, ensure_ascii=False)[
                    :ARGUMENTS_PREVIEW
                ],
                refused=c.refused,
                held=c.held,
                wrote=c.wrote,
                failed=c.failed,
                result_preview=c.result_preview,
            )
            for c in raw.tool_calls
        ],
    )


def _observed(run: _RawRun | None) -> str | None:
    """What Otto did in the first turn, the way a case's behaviour is named."""
    if run is None or not run.turns:
        return None
    first = run.turns[0]
    calls = [c for c in first.tool_calls if not c.refused]
    names = {c.name for c in calls}
    if first.ended_by == "finished":
        return "finish_task"
    if "delegate_to_expert" in names or any(
        c.name in ("start_task", "create_task") and c.arguments.get("expert_id")
        for c in calls
    ):
        return "delegate"
    if "start_task" in names:
        return "start_task"
    if first.ended_by == "question":
        return "ask"
    if "connect_integration" in names:
        return "sign_in"
    if names & {"update_task", "archive_task"}:
        return "task_update"
    return "reply"


def _output(output: dict[str, Any]) -> str | None:
    if not output:
        return None
    return json.dumps(output, indent=2, ensure_ascii=False)[:OUTPUT_PREVIEW]
