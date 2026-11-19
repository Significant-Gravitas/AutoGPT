"""Play one feature case: build what it needs, run it, check it, and have
each reviewer judge it when the case asks for a judgement."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from backend.copilot.eval.tasks.judge import review, run_checks
from backend.copilot.eval.tasks.llm import Backend
from backend.copilot.eval.tasks.models import CheckResult, Review, Run, World
from backend.copilot.eval.tasks.runner import Models, context_prefix

from .controls import brief_checks, settle_table, worker_case, worker_extra_checks
from .judge import SUGGESTIONS_PROMPT, rate, review_payload
from .live import TaskApi, run_live
from .models import FeatureCase, FeatureResult
from .records import make_task
from .suggestion_checks import suggestion_checks
from .suggestion_run import Drafter, draft_for

Engine = Callable[..., Awaitable[Run]]


class Settings:
    """How a run is set up: worlds, engines, reviewers and what may run."""

    def __init__(
        self,
        worlds: dict[str, World],
        models: Models | None = None,
        engine: Engine | None = None,
        reviewers: list[tuple[str, Backend, str]] | None = None,
        drafter: Drafter | None = None,
        live_api: TaskApi | None = None,
        live_user_id: str | None = None,
    ) -> None:
        self.worlds, self.models, self.engine = worlds, models, engine
        self.reviewers = reviewers or []
        self.drafter = drafter or Drafter()
        self.live_api, self.live_user_id = live_api, live_user_id

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.reviewers] or ["code"]


def skip_reason(case: FeatureCase, settings: Settings) -> str | None:
    if case.needs == "live" and (
        settings.live_api is None or not settings.live_user_id
    ):
        return "needs the live database, Redis, RabbitMQ and a copilot executor: run with --live --live-user-id"
    if (
        case.needs == "model"
        and settings.models is None
        and case.scripted_reply is None
    ):
        return "needs a model: run without --offline"
    return None


async def play(case: FeatureCase, settings: Settings) -> FeatureResult:
    base = FeatureResult(
        case_id=case.id,
        area=case.area,
        title=case.title,
        world=case.world,
        needs=case.needs,
    )
    if reason := skip_reason(case, settings):
        return base.model_copy(update={"skipped": reason})
    try:
        checks, shown, run, prompt = await _PLAYERS[case.area](case, settings)
    except Exception as e:  # a broken case is scored, not fatal to the run
        checks = [
            CheckResult(
                name="case_ran", passed=False, detail=f"{type(e).__name__}: {e}"[:500]
            )
        ]
        shown, run, prompt = {}, None, None
        base = base.model_copy(update={"error": checks[0].detail})
    reviews = await _reviews(case, settings, checks, shown, run, prompt)
    ratings = {name: rate(checks, r) for name, r in zip(settings.names, reviews)}
    return base.model_copy(
        update={
            "checks": checks,
            "ratings": ratings,
            "reviewed": any(r is not None for r in reviews),
            "output": shown,
            "run": run,
            "prompt_tokens": run.prompt_tokens if run else 0,
            "completion_tokens": run.completion_tokens if run else 0,
        }
    )


_Played = tuple[list[CheckResult], dict[str, Any], Run | None, str | None]


async def _suggestions(case: FeatureCase, settings: Settings) -> _Played:
    if case.account is None:
        raise ValueError(f"{case.id}: a suggestions case needs an account")
    drafter = settings.drafter
    if case.scripted_reply is not None:
        drafter = Drafter(scripted=case.scripted_reply)
    world = settings.worlds[case.world]
    drafting = await draft_for(case.account, world.experts, drafter)
    checks = suggestion_checks(case.expected.checks, drafting)
    shown = {
        "facts": drafting.calls[0].facts if drafting.calls else "(no model call)",
        "drafted": [
            {
                "title": t.title,
                "goal": t.goal,
                "done_when": t.done_when,
                "brief": t.description,
                "reason": t.suggested_reason,
                "owner": drafting.experts.get(t.owner_expert_id or "", None),
            }
            for t in drafting.created
        ],
        "live_after": [t.title for t in drafting.returned],
        "set_aside": drafting.dismissed,
    }
    return checks, shown, None, SUGGESTIONS_PROMPT


async def _controls(case: FeatureCase, settings: Settings) -> _Played:
    checks_spec = case.expected.checks
    if case.needs == "live":
        return await _live(case, settings)
    if checks_spec.settles_every_ending:
        return settle_table(), {}, None, None
    if case.task is None:
        raise ValueError(f"{case.id}: a controls case needs a task")
    task = make_task(case.id, case.task)
    if case.needs == "none":
        return brief_checks(checks_spec, task, case.note), {}, None, None
    if settings.engine is None or settings.models is None:
        raise RuntimeError("no worker engine configured")
    played = worker_case(case, task)
    run = await settings.engine(played, settings.worlds[case.world], settings.models)
    checks = run_checks(played, run) + worker_extra_checks(checks_spec, run)
    return checks, {}, run, "worker"


async def _live(case: FeatureCase, settings: Settings) -> _Played:
    if settings.live_api is None or not settings.live_user_id:
        raise RuntimeError("live tier not configured")
    checks, record = await run_live(case, settings.live_api, settings.live_user_id)
    shown: dict[str, Any] = (
        {"status": record.task.status.value} if record is not None else {}
    )
    return checks, shown, None, None


async def _reviews(
    case: FeatureCase,
    settings: Settings,
    checks: list[CheckResult],
    shown: dict[str, Any],
    run: Run | None,
    prompt: str | None,
) -> list[Review | None]:
    if not case.review or not settings.reviewers or prompt is None:
        return [None for _ in settings.names]
    if prompt == "worker" and run is not None:
        world = settings.worlds[case.world]
        played = worker_case(case, make_task(case.id, case.task)) if case.task else None
        if played is None:
            return [None for _ in settings.names]
        knew = context_prefix(world, world.approval_mode)
        jobs = [
            review(played, run, checks, b, m, knew) for _, b, m in settings.reviewers
        ]
    else:
        jobs = [
            review_payload(prompt, case, shown, checks, b, m)
            for _, b, m in settings.reviewers
        ]
    return list(await asyncio.gather(*jobs))


_PLAYERS = {
    "suggestions": _suggestions,
    "controls": _controls,
}
