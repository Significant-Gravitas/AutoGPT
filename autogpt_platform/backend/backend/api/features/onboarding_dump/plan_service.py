"""The plan step's reads and the hire that follows the trial.

``get_plan`` answers the onboarding plan step: what Otto heard, the team of
at most three, and the tasks in their ranked order with their owners. A
brain dump the greeting wrote tasks from gives the plan it stored
(``plan_job``); a skipped, failed or empty one gives the role plan
(``role_tasks``), stored on first read so its ids stay the ones Home uses.

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
    RecommendedExpert,
    SuggestedPrompt,
    SuggestedTask,
)
from backend.api.features.onboarding_dump.plan_models import (
    PlanHiredExpert,
    PlanHireFailure,
    PlanHireResponse,
    PlanResponse,
)
from backend.util.exceptions import ExpertLimitExceededError

logger = logging.getLogger(__name__)

MAX_HEARD_CHARS = 160
_TERMINAL = (BrainDumpStatus.completed, BrainDumpStatus.failed)
# Sentences that ask the user to act rather than say what Otto heard.
_INVITATION = re.compile(
    r"^(pick|choose|here are|here['’]s|tell me|let['’]s|let me|want|start"
    r"|where|what|which|ready|just)\b",
    re.IGNORECASE,
)


async def get_plan(user_id: str) -> PlanResponse:
    """The plan for the plan step; ``ready=false`` only while the pipeline runs.

    While it runs the answer carries the role plan, unstored, so a client
    that stops waiting still has a true plan to show.
    """
    dump = await db.get_dump(user_id)
    if dump is not None and _pending(dump):
        prompts, team = await _role_plan(user_id)
        return await _response(
            user_id,
            suggested_tasks.as_tasks(prompts),
            team,
            ready=False,
            source="fallback",
            skipped_dump=False,
        )
    skipped = dump is None or service.nothing_to_reflect(dump)
    stored = suggested_tasks.stored_tasks(dump.suggestedPrompts if dump else None)
    if dump is not None and not skipped and (dump.greeting or "").strip() and stored:
        # The greeting wrote these, or the plan job swapped its generic
        # starters for the role set; either way the pipeline's team owns them.
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


async def hire_plan(user_id: str) -> PlanHireResponse:
    """Hire the plan's team, skipping anyone already on it.

    A second call hires nobody and reports the same people. One failed hire
    is reported and the rest go ahead; the active expert limit stops the
    hiring there, and everyone not yet hired is reported with it.
    """
    team = (await get_plan(user_id)).team
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
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", greeting.strip()) if s]
    while sentences and (
        sentences[-1].endswith("?") or _INVITATION.match(sentences[-1])
    ):
        sentences.pop()
    if not sentences:
        return None
    two = " ".join(sentences[:2])
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
    none (its write failed), from reads that raise like the role plan's."""
    team = plan_job.stored_team(dump.recommendedExperts)
    if team is not None and team.source != "disabled":
        return team.experts
    return (await _role_plan(user_id))[1]


def _pending(dump: OnboardingBrainDump) -> bool:
    """Whether the pipeline is still working on ``dump``.

    A voice take still being transcribed has no transcript yet, so it would
    read as skipped; it is pending like any take the pipeline has.
    """
    if dump.status == BrainDumpStatus.transcribing:
        return True
    return not service.nothing_to_reflect(dump) and dump.status not in _TERMINAL


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
