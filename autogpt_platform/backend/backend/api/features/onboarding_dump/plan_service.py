"""The plan step's reads and the hire that follows the trial.

``get_plan`` answers the onboarding plan step: what Otto heard, the team of
at most three, and the tasks in their ranked order with their owners. A
brain dump the greeting wrote tasks from gives the plan it stored
(``plan_job``); a skipped, failed or empty one gives the role plan
(``role_tasks``), stored on first read so its ids stay the ones Home uses.

``settle_plan`` is the step giving up on the wait: while the pipeline is
still running, the role plan the step shows becomes the take's plan, so the
hire and Home use what the step showed.

``hire_plan`` hires that team once the trial is confirmed. It is idempotent
and never fails the request for one bad template: whoever could not join is
reported, and their tasks read as Otto's on Home.
"""

import logging
import re
from typing import Literal

from prisma.enums import BrainDumpStatus
from prisma.models import OnboardingBrainDump

from backend.api.features.experts import experts_db
from backend.api.features.experts.experts_db import list_templates
from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import (
    db,
    plan_job,
    plan_owners,
    role_tasks,
    service,
    suggested_tasks,
)
from backend.api.features.onboarding_dump.models import (
    ExpertRecommendations,
    RecommendedExpert,
    SuggestedPrompt,
    SuggestedTask,
)
from backend.api.features.onboarding_dump.plan_models import (
    PlanHiredExpert,
    PlanHireFailure,
    PlanHireResponse,
    PlanProgress,
    PlanResponse,
)
from backend.util.exceptions import ExpertLimitExceededError

logger = logging.getLogger(__name__)

MAX_HEARD_CHARS = 160
_TERMINAL = (BrainDumpStatus.completed, BrainDumpStatus.failed)
# Each attempt is one read and one conditional write, as for the plan job.
_SETTLE_ATTEMPTS = 3
# Sentences that ask the user to act rather than say what Otto heard.
_INVITATION = re.compile(
    r"^(pick|choose|here are|here['’]s|tell me|let['’]s|let me|want|start"
    r"|where|what|which|ready|just)\b",
    re.IGNORECASE,
)


async def get_plan(user_id: str) -> PlanResponse:
    """The plan for the plan step; ``ready=false`` only while the pipeline runs.

    While it runs the answer carries the role plan, unstored, so a client
    that stops waiting still has a true plan to show, and how far the jobs
    have got. A take whose tasks are already a plan (the plan job ranked
    them, or the step settled on the role plan) is ready at once.
    """
    dump = await db.get_dump(user_id)
    # A plan stored while the jobs still run: the step settled on the role
    # plan, or the plan job wrote its own a moment before the completion.
    early = dump is not None and _pending(dump) and _has_plan(dump)
    if dump is not None and _pending(dump) and not early:
        prompts, team = await _role_plan(user_id)
        response = await _response(
            user_id,
            suggested_tasks.as_tasks(prompts),
            team,
            ready=False,
            source="fallback",
            skipped_dump=False,
        )
        response.progress = PlanProgress(
            tasks_written=bool((dump.greeting or "").strip()),
            team_chosen=dump.recommendedExperts is not None,
        )
        return response
    skipped = dump is None or service.nothing_to_reflect(dump)
    stored = suggested_tasks.stored_tasks(dump.suggestedPrompts if dump else None)
    if (
        dump is not None
        and not skipped
        and stored
        and ((dump.greeting or "").strip() or early)
    ):
        # The greeting wrote these, the plan job swapped its generic starters
        # for the role set, or the step settled on the role plan; either way
        # the team stored with them owns them.
        wrote_tasks = not role_tasks.is_default_set(stored)
        return await _response(
            user_id,
            stored,
            await _pipeline_team(user_id, dump),
            ready=True,
            source="llm" if wrote_tasks else "fallback",
            skipped_dump=False,
            heard=heard_from(dump.greeting or "") if wrote_tasks else None,
        )
    prompts, team = await _role_plan(user_id)
    tasks = stored or await suggested_tasks.store_if_empty(user_id, prompts)
    return await _response(
        user_id, tasks, team, ready=True, source="fallback", skipped_dump=skipped
    )


async def settle_plan(user_id: str) -> PlanResponse:
    """The plan the step shows once it stops waiting, made the take's plan.

    While the pipeline is still running and has not written a plan, the
    role plan the step was handed is stored as the take's tasks and team:
    the greeting and team jobs then leave both alone (``db.store_greeting``,
    ``service._store_team``) and the plan job finds nothing to rank, so the
    trial hires, and Home lists, what the step showed. Otherwise there is
    nothing to settle, and this answers like ``get_plan``.
    """
    for _ in range(_SETTLE_ATTEMPTS):
        dump = await db.get_dump(user_id)
        if dump is None or not _pending(dump) or _has_plan(dump):
            break
        prompts, team = await _role_plan(user_id)
        tasks, capped = plan_owners.cap_team(suggested_tasks.as_tasks(prompts), team)
        if await db.replace_plan(
            user_id,
            dump.recordingId,
            dump.updatedAt,
            [suggested_tasks.storable(task) for task in tasks],
            ExpertRecommendations(experts=capped).model_dump(mode="json"),
        ):
            logger.info(
                f"Brain dump plan for user {user_id} settled on the role plan: "
                f"{len(tasks)} tasks, {len(capped)} experts"
            )
            break
    return await get_plan(user_id)


async def hire_plan(user_id: str) -> PlanHireResponse:
    """Hire the plan's team, skipping anyone already on it.

    A second call hires nobody and reports the same people. One failed hire
    is reported and the rest go ahead; the active expert limit stops the
    hiring there, and everyone not yet hired is reported with it.
    """
    # Settled first: a team hired while the pipeline still runs is the one
    # whose tasks Home then lists, not a team a later plan would replace.
    team = (await settle_plan(user_id)).team
    active = _on_team(await suggested_tasks.roster_by_template(user_id))
    hired: list[PlanHiredExpert] = []
    failed: list[PlanHireFailure] = []
    for position, expert in enumerate(team):
        if expert.template_id in active:
            hired.append(active[expert.template_id])
            continue
        try:
            result = await experts_db.hire_expert(
                user_id, expert.template_id, None, surface="onboarding", via="plan"
            )
        except ExpertLimitExceededError:
            # Nobody else can join; teammates already on the team still are.
            for rest in team[position:]:
                if rest.template_id in active:
                    hired.append(active[rest.template_id])
                else:
                    failed.append(_failure(rest))
            break
        except Exception as e:
            logger.warning(f"Plan hire of {expert.template_id} failed: {e}")
            failed.append(_failure(expert))
            continue
        hired.append(
            PlanHiredExpert(
                template_id=expert.template_id,
                expert_id=result.expert.id,
                name=result.expert.name,
            )
        )
    logger.info(
        f"Brain dump plan hire for user {user_id}: {len(team)} in the team, "
        f"{len(hired)} on it now ({len(active)} before), {len(failed)} failed"
    )
    return PlanHireResponse(hired=hired, failed=failed)


def heard_from(greeting: str) -> str | None:
    """The first two sentences of the greeting, without its invitation,
    in at most ``MAX_HEARD_CHARS``."""
    # Western sentences end at punctuation and a space; Chinese and Japanese
    # ones at a full-width mark with no space after it.
    sentences = [
        s for s in re.split(r"(?<=[.!?…])\s+|(?<=[。！？])", greeting.strip()) if s
    ]
    while sentences and (
        sentences[-1].endswith(("?", "？")) or _INVITATION.match(sentences[-1])
    ):
        sentences.pop()
    if not sentences:
        return None
    two = _join(sentences[:2])
    if len(two) <= MAX_HEARD_CHARS:
        return two
    first = sentences[0]
    if len(first) <= MAX_HEARD_CHARS:
        return first
    cut = first[: MAX_HEARD_CHARS - 1]
    # Back to the last whole word, when there is one to go back to.
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(",;:—-") + "…"


async def _response(
    user_id: str,
    tasks: list[SuggestedTask],
    team: list[RecommendedExpert],
    *,
    ready: bool,
    source: Literal["llm", "fallback"],
    skipped_dump: bool,
    heard: str | None = None,
) -> PlanResponse:
    # Re-applied on every read: a row written before owners existed, or by
    # a pipeline that never capped it, still shows at most three people.
    tasks, capped = plan_owners.cap_team(tasks, team)
    return PlanResponse(
        ready=ready,
        source=source,
        skipped_dump=skipped_dump,
        heard=heard,
        team=capped,
        tasks=await suggested_tasks.with_expert_ids(user_id, tasks),
    )


async def _pipeline_team(
    user_id: str, dump: OnboardingBrainDump
) -> list[RecommendedExpert]:
    """The team the dump's team job stored, or the role team when it stored
    none (its write failed, or a failed roster read left it empty), from
    reads that raise like the role plan's."""
    team = plan_job.stored_team(dump.recommendedExperts)
    if team is not None and team.source != "disabled" and _has_answer(team):
        return team.experts
    return (await _role_plan(user_id))[1]


def _has_answer(team: ExpertRecommendations) -> bool:
    # No experts and no expert to raise is what a roster that could not be
    # read leaves; a real "nobody fits" still suggests raising one.
    return bool(team.experts) or team.raise_suggestion is not None


def _join(sentences: list[str]) -> str:
    """Two sentences as one line: a space between them, except after a
    full-width mark, which carries its own."""
    first, *rest = sentences
    return first + "".join(
        ("" if first.endswith(("。", "！", "？")) else " ") + s for s in rest
    )


def _pending(dump: OnboardingBrainDump) -> bool:
    """Whether the pipeline is still working on ``dump``.

    A voice take still being transcribed has no transcript yet, so it would
    read as skipped; it is pending like any take the pipeline has.
    """
    if dump.status == BrainDumpStatus.transcribing:
        return True
    return not service.nothing_to_reflect(dump) and dump.status not in _TERMINAL


def _has_plan(dump: OnboardingBrainDump) -> bool:
    """Whether the take's tasks are already a plan, each with the id Home
    uses: the plan job ranked them, or a reader stored them. The greeting's
    own rows carry no id."""
    raw = dump.suggestedPrompts
    return isinstance(raw, list) and any(
        isinstance(row, dict) and "id" in row for row in raw
    )


async def _role_plan(
    user_id: str,
) -> tuple[list[SuggestedPrompt], list[RecommendedExpert]]:
    """The role plan, from reads that raise: what a failed read gives would
    otherwise be stored, or hired, as the plan."""
    user_role, pain_points = await service.read_wizard_answers(user_id)
    return role_tasks.role_plan(user_role, pain_points, await list_templates())


def _on_team(roster: dict[str, Expert]) -> dict[str, PlanHiredExpert]:
    return {
        template_id: PlanHiredExpert(
            template_id=template_id, expert_id=expert.id, name=expert.name
        )
        for template_id, expert in roster.items()
    }


def _failure(expert: RecommendedExpert) -> PlanHireFailure:
    return PlanHireFailure(template_id=expert.template_id, name=expert.name)
