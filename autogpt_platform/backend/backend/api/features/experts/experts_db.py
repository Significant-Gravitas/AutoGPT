import logging

import prisma.errors
import prisma.models

from backend.api.features.experts import scheduling
from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    ExpertSoulUpdate,
    ExpertWorkflowRef,
    HireResult,
)
from backend.api.features.library import db as library_db
from backend.api.features.orgs.db import get_user_default_team
from backend.data.expert_spend import get_weekly_spend
from backend.data.user import get_user_by_id
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)

_WORKFLOW_ROW_INCLUDE = {"LibraryAgent": True, "StoreListingVersion": True}
_WORKFLOW_INCLUDE = {"Workflows": {"include": _WORKFLOW_ROW_INCLUDE}}


class ExpertTemplateNotFoundError(Exception):
    def __init__(self, template_id: str):
        super().__init__(f"Expert template {template_id} not found")
        self.template_id = template_id


class ExpertNotFoundError(Exception):
    def __init__(self, expert_id: str):
        super().__init__(f"Expert {expert_id} not found")
        self.expert_id = expert_id


class ExpertPersonalTenancyNotFoundError(Exception):
    def __init__(self, expert_id: str):
        super().__init__(f"Personal tenancy for expert {expert_id} not found")
        self.expert_id = expert_id


def _to_workflow_ref(row: prisma.models.ExpertWorkflow) -> ExpertWorkflowRef:
    listing = row.StoreListingVersion
    library_agent = row.LibraryAgent
    return ExpertWorkflowRef(
        id=row.id,
        store_listing_version_id=row.storeListingVersionId,
        library_agent_id=row.libraryAgentId,
        graph_id=library_agent.agentGraphId if library_agent else None,
        name=listing.name if listing else None,
        description=listing.description if listing else None,
        schedule_cron=row.scheduleCron,
        schedule_id=row.scheduleId,
    )


def _to_model(
    row: prisma.models.Expert,
    latest_run: prisma.models.AgentGraphExecution | None = None,
    weekly_spend: int = 0,
) -> Expert:
    return Expert(
        id=row.id,
        name=row.name,
        avatar_url=row.avatarUrl,
        role=row.role,
        tagline=row.tagline,
        bio=row.bio,
        skills=row.skills or [],
        identity=row.identity,
        voice_preferences=row.voicePreferences,
        boundaries=row.boundaries,
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=row.isTemplate,
        source_template_id=row.sourceTemplateId,
        is_archived=row.isArchived,
        workflows=[_to_workflow_ref(w) for w in row.Workflows or []],
        last_run_at=latest_run.createdAt if latest_run else None,
        last_run_status=(str(latest_run.executionStatus) if latest_run else None),
        weekly_budget=scheduling.effective_weekly_budget(row),
        weekly_spend=weekly_spend,
        schedules_paused_at=row.schedulesPausedAt,
    )


async def _latest_runs(
    expert_ids: list[str],
) -> dict[str, prisma.models.AgentGraphExecution]:
    """Latest execution per expert, one indexed query via Prisma distinct."""
    if not expert_ids:
        return {}
    rows = await prisma.models.AgentGraphExecution.prisma().find_many(
        where={"expertId": {"in": expert_ids}, "isDeleted": False},
        order=[{"expertId": "asc"}, {"createdAt": "desc"}],
        distinct=["expertId"],
    )
    return {row.expertId: row for row in rows if row.expertId is not None}


async def list_templates() -> list[Expert]:
    rows = await prisma.models.Expert.prisma().find_many(
        where={"isTemplate": True, "isArchived": False},
        include=_WORKFLOW_INCLUDE,
    )
    return [_to_model(row) for row in rows]


async def list_experts(user_id: str) -> list[Expert]:
    rows = await prisma.models.Expert.prisma().find_many(
        where={
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        },
        include=_WORKFLOW_INCLUDE,
    )
    latest_runs = await _latest_runs([row.id for row in rows])
    return [
        _to_model(row, latest_runs.get(row.id), await get_weekly_spend(row.id))
        for row in rows
    ]


async def get_expert(
    user_id: str, expert_id: str, *, include_workflows: bool = True
) -> Expert | None:
    """Fetch a hired expert owned by *user_id*.

    Set ``include_workflows=False`` to skip the ExpertWorkflow → LibraryAgent
    + StoreListingVersion joins when the caller only needs the expert's own
    columns. The returned model then always carries an empty ``workflows``
    list — never use that flag to decide whether workflows are installed.
    """
    row = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        },
        include=_WORKFLOW_INCLUDE if include_workflows else None,
    )
    if row is None:
        return None
    latest_runs = await _latest_runs([row.id])
    return _to_model(row, latest_runs.get(row.id), await get_weekly_spend(row.id))


async def resolve_expert_personal_tenancy(
    user_id: str, expert_id: str
) -> tuple[str, str | None]:
    """Return the personal org and default team for an active owned expert."""
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        }
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)

    organization_id, team_id = await get_user_default_team(user_id)
    if organization_id is None:
        raise ExpertPersonalTenancyNotFoundError(expert_id)
    return organization_id, team_id


async def hire_expert(user_id: str, template_id: str, name: str | None) -> HireResult:
    template = await prisma.models.Expert.prisma().find_first(
        where={"id": template_id, "isTemplate": True, "isArchived": False},
        include=_WORKFLOW_INCLUDE,
    )
    if template is None:
        raise ExpertTemplateNotFoundError(template_id)

    existing = await prisma.models.Expert.prisma().find_first(
        where={"ownerUserId": user_id, "sourceTemplateId": template_id},
        include=_WORKFLOW_INCLUDE,
    )
    if existing is not None:
        return await _existing_hire_result(existing)

    create_data: dict = {
        "ownerUserId": user_id,
        "name": name or template.name,
        "avatarUrl": template.avatarUrl,
        "role": template.role,
        "tagline": template.tagline,
        "bio": template.bio,
        "skills": template.skills or [],
        "identity": template.identity,
        "voicePreferences": template.voicePreferences,
        "boundaries": template.boundaries,
        "sourceTemplateId": template.id,
    }
    if template.toolProfile is not None:
        create_data["toolProfile"] = template.toolProfile
    try:
        expert = await prisma.models.Expert.prisma().create(data=create_data)
    except prisma.errors.UniqueViolationError:
        # Lost a concurrent hire race; the winner's row satisfies idempotency.
        raced = await prisma.models.Expert.prisma().find_first(
            where={"ownerUserId": user_id, "sourceTemplateId": template_id},
            include=_WORKFLOW_INCLUDE,
        )
        if raced is None:
            raise
        return await _existing_hire_result(raced)

    failed = await _install_preloads(expert.id, user_id, template.Workflows or [])

    hydrated = await prisma.models.Expert.prisma().find_unique(
        where={"id": expert.id}, include=_WORKFLOW_INCLUDE
    )
    if hydrated is None:
        raise ExpertNotFoundError(expert.id)
    return HireResult(expert=_to_model(hydrated), failed_preloads=failed)


async def _existing_hire_result(row: prisma.models.Expert) -> HireResult:
    """Idempotent-hire result for an already-existing hired copy.

    Re-hiring an archived expert revives it — the unique
    (ownerUserId, sourceTemplateId) constraint means a fresh row cannot be
    created, and returning the archived row as-is would hand back a
    "successful" hire that stays invisible to list_experts/get_expert.
    """
    if row.isArchived:
        revived = await prisma.models.Expert.prisma().update(
            where={"id": row.id},
            data={"isArchived": False},
            include=_WORKFLOW_INCLUDE,
        )
        if revived is not None:
            row = revived
        if row.ownerUserId:
            await scheduling.resume_expert_schedules(row.ownerUserId, row.id)
            try:
                await scheduling.reattach_expert_triggers(row.ownerUserId, row.id)
            except Exception:
                logger.exception(
                    f"Failed to reattach triggers while reviving expert #{row.id}"
                )
            # Resume/reattach mutated pause state and workflow scheduleIds
            # after `row` was read — reload so the result isn't stale.
            refreshed = await prisma.models.Expert.prisma().find_unique(
                where={"id": row.id}, include=_WORKFLOW_INCLUDE
            )
            if refreshed is not None:
                row = refreshed
    return HireResult(expert=_to_model(row), failed_preloads=[])


async def update_soul(user_id: str, expert_id: str, soul: ExpertSoulUpdate) -> Expert:
    updated = await prisma.models.Expert.prisma().update_many(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        },
        data={
            "name": soul.name,
            "identity": soul.identity,
            "voicePreferences": soul.voice_preferences,
            "boundaries": soul.boundaries,
        },
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)

    expert = await get_expert(user_id, expert_id)
    if expert is None:
        raise ExpertNotFoundError(expert_id)
    return expert


async def _install_preloads(
    expert_id: str, user_id: str, preloads: list[prisma.models.ExpertWorkflow]
) -> list[str]:
    """Install template preloads into the hiring user's library.

    Honest partial hire: a failed preload is logged and reported, never
    fatal to the hire itself. Preloads with a roster cadence also get their
    schedule created here (see ``_schedule_preload``).
    """
    failed: list[str] = []
    user_timezone: str | None = None
    if any(p.scheduleCron for p in preloads):
        user = await get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)
    for preload in preloads:
        if preload.storeListingVersionId is None:
            continue
        try:
            library_agent = await library_db.add_store_agent_to_library(
                preload.storeListingVersionId, user_id
            )
            row = await prisma.models.ExpertWorkflow.prisma().create(
                data={
                    "expertId": expert_id,
                    "storeListingVersionId": preload.storeListingVersionId,
                    "libraryAgentId": library_agent.id,
                    "scheduleCron": preload.scheduleCron,
                }
            )
        except Exception:
            logger.exception(
                f"Failed to install preload {preload.storeListingVersionId} "
                f"on expert #{expert_id} for user #{user_id}"
            )
            failed.append(
                preload.StoreListingVersion.name
                if preload.StoreListingVersion
                else preload.storeListingVersionId
            )
            continue
        if preload.scheduleCron:
            listing = preload.StoreListingVersion
            await scheduling.create_workflow_schedule(
                workflow_row_id=row.id,
                expert_id=expert_id,
                user_id=user_id,
                cron=preload.scheduleCron,
                graph_id=library_agent.graph_id,
                graph_version=library_agent.graph_version,
                name=listing.name if listing else "Expert workflow",
                user_timezone=user_timezone or "UTC",
            )
    return failed


async def install_workflow(
    user_id: str, expert_id: str, store_listing_version_id: str
) -> ExpertWorkflowRef:
    expert = await prisma.models.Expert.prisma().find_first(
        where={
            "id": expert_id,
            "ownerUserId": user_id,
            "isTemplate": False,
            "isArchived": False,
        }
    )
    if expert is None:
        raise ExpertNotFoundError(expert_id)

    existing = await prisma.models.ExpertWorkflow.prisma().find_first(
        where={
            "expertId": expert_id,
            "storeListingVersionId": store_listing_version_id,
        },
        include=_WORKFLOW_ROW_INCLUDE,
    )
    if existing is not None:
        return _to_workflow_ref(existing)

    library_agent = await library_db.add_store_agent_to_library(
        store_listing_version_id, user_id
    )
    try:
        row = await prisma.models.ExpertWorkflow.prisma().create(
            data={
                "expertId": expert_id,
                "storeListingVersionId": store_listing_version_id,
                "libraryAgentId": library_agent.id,
            },
            include=_WORKFLOW_ROW_INCLUDE,
        )
    except prisma.errors.UniqueViolationError:
        # Lost a concurrent duplicate-install race; return the winner's row.
        raced = await prisma.models.ExpertWorkflow.prisma().find_first(
            where={
                "expertId": expert_id,
                "storeListingVersionId": store_listing_version_id,
            },
            include=_WORKFLOW_ROW_INCLUDE,
        )
        if raced is None:
            raise
        return _to_workflow_ref(raced)
    return _to_workflow_ref(row)


async def resolve_expert_for_graph(user_id: str, graph_id: str) -> str | None:
    """Expert attribution for a manually scheduled graph.

    Returns the id of the single active hired expert that has *graph_id*
    installed as a workflow. Two experts can install the same listing and
    share one LibraryAgent, which makes the join ambiguous — on anything
    but a unique match this declines (returns ``None``) rather than guess.
    """
    rows = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={
            "Expert": {
                "is": {
                    "ownerUserId": user_id,
                    "isTemplate": False,
                    "isArchived": False,
                }
            },
            "LibraryAgent": {
                "is": {
                    "userId": user_id,
                    "agentGraphId": graph_id,
                    "isDeleted": False,
                }
            },
        }
    )
    expert_ids = {row.expertId for row in rows}
    if len(expert_ids) != 1:
        return None
    return expert_ids.pop()


async def archive_expert(user_id: str, expert_id: str) -> None:
    updated = await prisma.models.Expert.prisma().update_many(
        where={"id": expert_id, "ownerUserId": user_id, "isTemplate": False},
        data={"isArchived": True},
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)
    await scheduling.pause_expert_schedules(
        user_id, expert_id, reason="Expert archived"
    )
    try:
        await scheduling.detach_expert_triggers(user_id, expert_id)
    except Exception:
        # The archive itself must not fail on a scheduler hiccup: presets
        # are already deactivated first inside detach, and the run-time
        # gate refuses archived experts as the backstop.
        logger.exception(
            f"Failed to detach triggers while archiving expert #{expert_id}"
        )
