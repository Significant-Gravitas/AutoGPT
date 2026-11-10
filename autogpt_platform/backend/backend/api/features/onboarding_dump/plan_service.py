"""The plan step's reads, and the plan it settles on when it stops waiting.

``get_plan`` answers the onboarding plan step: what Otto heard, the team of
at most three, and the tasks in their ranked order with their owners. A
brain dump the greeting wrote tasks from gives the plan it stored
(``plan_job``); a skipped, failed or empty one gives the role plan
(``role_tasks``), stored on first read so its ids stay the ones Home uses.

``settle_plan`` is the step giving up on the wait: while the pipeline is
still running, the plan the step shows becomes the take's plan, so the hire
(``plan_hire``) and Home use what the step showed.
"""

import logging
import re
from typing import Literal

from prisma.enums import BrainDumpStatus
from prisma.models import OnboardingBrainDump

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
from backend.api.features.onboarding_dump.plan_models import PlanProgress, PlanResponse
from backend.util.cache import cached

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
    them, or the step settled on them) is ready at once.
    """
    dump = await db.get_dump(user_id)
    # A plan stored while the jobs still run: the step settled on one, or
    # the plan job wrote its own a moment before the completion.
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
    # A take settled while still being transcribed has no transcript yet,
    # but it was recorded, not skipped.
    skipped = dump is None or (service.nothing_to_reflect(dump) and not early)
    stored = suggested_tasks.stored_tasks(dump.suggestedPrompts if dump else None)
    if (
        dump is not None
        and not skipped
        and stored
        and ((dump.greeting or "").strip() or early)
    ):
        # The greeting wrote these, the plan job swapped its generic starters
        # for the role set, or the step settled on them or the role plan;
        # either way the team stored with them owns them.
        wrote_tasks = not role_tasks.is_default_set(stored)
        return await _response(
            user_id,
            stored,
            (await _pipeline_team(user_id, dump)).experts,
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

    While the pipeline is still running and has not written a plan, what
    it has so far becomes the take's tasks and team (``_settled_plan``):
    the greeting and team jobs then leave both alone (``db.store_greeting``,
    ``service._store_team``) and the plan job finds nothing to rank, so the
    trial hires, and Home lists, what the step showed. Otherwise there is
    nothing to settle, and this answers like ``get_plan``.
    """
    for _ in range(_SETTLE_ATTEMPTS):
        dump = await db.get_dump(user_id)
        if dump is None or not _pending(dump) or _has_plan(dump):
            break
        prompts, team = await _settled_plan(user_id, dump)
        tasks, capped = plan_owners.cap_team(
            suggested_tasks.as_tasks(prompts), team.experts
        )
        if await db.replace_plan(
            user_id,
            dump.recordingId,
            dump.updatedAt,
            [suggested_tasks.storable(task) for task in tasks],
            team.model_copy(update={"experts": capped}).model_dump(mode="json"),
        ):
            logger.info(
                f"Brain dump plan for user {user_id} settled: {len(tasks)} tasks, "
                f"{len(capped)} experts, "
                f"greeting's tasks={not role_tasks.is_default_set(prompts)}"
            )
            break
    return await get_plan(user_id)


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
        # Onboarding names owners from ``team``: a roster blip must not fail
        # the plan, or the hire that settles it.
        tasks=await suggested_tasks.with_expert_ids(
            user_id, tasks, roster_required=False
        ),
    )


async def _settled_plan(
    user_id: str, dump: OnboardingBrainDump
) -> tuple[list[SuggestedPrompt], ExpertRecommendations]:
    """What settling stores: the tasks the greeting has written, owned as the
    plan job owns them without Jev, with the dump's team; the role plan while
    there are none, or when the greeting fell back to the generic starters."""
    written = plan_job.rankable_prompts(dump.suggestedPrompts)
    if not written or role_tasks.is_default_set(written):
        prompts, team = await _role_plan(user_id)
        return prompts, ExpertRecommendations(experts=team)
    team = await _pipeline_team(user_id, dump)
    return plan_owners.fallback_assign(written, team.experts), team


async def _pipeline_team(
    user_id: str, dump: OnboardingBrainDump
) -> ExpertRecommendations:
    """The team the dump's team job stored, or the role team when it stored
    none (it is still choosing, its write failed, or a failed roster read
    left it empty), from reads that raise like the role plan's."""
    team = plan_job.stored_team(dump.recommendedExperts)
    if team is not None and team.source != "disabled" and _has_answer(team):
        return team
    return ExpertRecommendations(experts=(await _role_plan(user_id))[1])


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
    return role_tasks.role_plan(user_role, pain_points, await _roster())


# The step reads the role plan every second while the pipeline runs; the
# roster changes only when a deploy publishes the catalog.
@cached(maxsize=1, ttl_seconds=60)
async def _roster() -> list[Expert]:
    return await list_templates()
