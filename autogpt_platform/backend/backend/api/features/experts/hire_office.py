"""Hire a whole office pack in one call.

An OfficeTemplate's config lists expert templates plus an intro task per
expert. Hiring the office reserves every expert copy AND opens every intro
DelegatedTask inside ONE transaction — a failure on the Nth expert rolls
the whole office back, so the user never ends up with half a team. Preload
installs and scheduler registration stay outside the transaction: both are
best-effort in the single-hire flow too, and the scheduler is an external
service that must not hold a DB transaction open.
"""

import logging

import prisma.enums
import prisma.models
from pydantic import BaseModel, Field

from backend.api.features.experts import experts_db, scheduling
from backend.api.features.experts.errors import ExpertTemplateNotFoundError
from backend.api.features.experts.models import Expert
from backend.api.features.tasks.models import (
    TASK_SPEC_MAX_LENGTH,
    TASK_TITLE_MAX_LENGTH,
)
from backend.data.db import transaction
from backend.data.user import get_user_by_id
from backend.util.exceptions import ExpertNotFoundError
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)


class OfficeTemplateNotFoundError(Exception):
    def __init__(self, office_template_id: str):
        super().__init__(f"Office template #{office_template_id} not found")


class OfficeExpertEntry(BaseModel):
    """One expert line in an OfficeTemplate config."""

    template_id: str
    schedule_cron: str | None = None
    intro_task_title: str
    intro_task_spec: str


class OfficeConfig(BaseModel):
    experts: list[OfficeExpertEntry] = Field(min_length=1)


class OfficeTemplateExpert(BaseModel):
    template_id: str
    name: str
    role: str
    avatar_url: str | None
    tagline: str | None
    schedule_cron: str | None
    intro_task_title: str


class OfficeTemplateSummary(BaseModel):
    id: str
    name: str
    description: str
    experts: list[OfficeTemplateExpert]


class HiredOfficeExpert(BaseModel):
    expert: Expert
    intro_task_id: str
    intro_task_title: str
    schedule_created: bool


class HireOfficeResult(BaseModel):
    office_template_id: str
    office_name: str
    hired: list[HiredOfficeExpert]


async def list_office_templates() -> list[OfficeTemplateSummary]:
    """Every office pack, with its expert lines joined to the template rows.

    A pack whose config no longer validates, or whose expert templates are
    gone, is skipped with a warning rather than failing the listing.
    """
    rows = await prisma.models.OfficeTemplate.prisma().find_many(order={"name": "asc"})
    summaries = []
    for row in rows:
        config = _parse_config(row)
        if config is None:
            continue
        templates = await _load_templates(config)
        if templates is None:
            logger.warning(f"Office template '{row.name}' references missing experts")
            continue
        summaries.append(
            OfficeTemplateSummary(
                id=row.id,
                name=row.name,
                description=row.description,
                experts=[
                    OfficeTemplateExpert(
                        template_id=entry.template_id,
                        name=templates[entry.template_id].name,
                        role=templates[entry.template_id].role,
                        avatar_url=templates[entry.template_id].avatarUrl,
                        tagline=templates[entry.template_id].tagline,
                        schedule_cron=entry.schedule_cron,
                        intro_task_title=entry.intro_task_title,
                    )
                    for entry in config.experts
                ],
            )
        )
    return summaries


async def hire_office(user_id: str, office_template_id: str) -> HireOfficeResult:
    office = await prisma.models.OfficeTemplate.prisma().find_unique(
        where={"id": office_template_id}
    )
    if office is None:
        raise OfficeTemplateNotFoundError(office_template_id)
    config = OfficeConfig.model_validate(office.config)
    templates = await _load_templates(config)
    if templates is None:
        raise ExpertTemplateNotFoundError(office_template_id)

    reserved = await _reserve_office(user_id, config, templates)

    hired = []
    for entry, expert_row, state, task_id in reserved:
        await _finish_hire(user_id, expert_row, state, templates[entry.template_id])
        schedule_created = False
        if entry.schedule_cron:
            schedule_created = await _create_office_schedule(
                user_id, expert_row.id, entry.schedule_cron
            )
        hired.append(
            HiredOfficeExpert(
                expert=await _hydrated_expert(expert_row.id),
                intro_task_id=task_id,
                intro_task_title=entry.intro_task_title,
                schedule_created=schedule_created,
            )
        )
    return HireOfficeResult(
        office_template_id=office.id,
        office_name=office.name,
        hired=hired,
    )


_ReservedEntry = tuple[OfficeExpertEntry, prisma.models.Expert, str, str]


async def _reserve_office(
    user_id: str,
    config: OfficeConfig,
    templates: dict[str, prisma.models.Expert],
) -> list[_ReservedEntry]:
    """All expert copies + intro tasks in ONE transaction, under the same
    per-user creation lock the single-hire flow takes."""
    reserved: list[_ReservedEntry] = []
    async with transaction() as tx:
        await experts_db.lock_expert_creation(tx, user_id)
        for entry in config.experts:
            template = templates[entry.template_id]
            expert_row, state = await experts_db.reserve_hired_expert_locked(
                tx,
                user_id,
                template.id,
                experts_db.hire_create_data(user_id, template),
            )
            task = await _create_intro_task(tx, user_id, expert_row.id, entry)
            reserved.append((entry, expert_row, state, task.id))
    return reserved


async def _create_intro_task(
    tx: prisma.Prisma,
    user_id: str,
    expert_id: str,
    entry: OfficeExpertEntry,
) -> prisma.models.DelegatedTask:
    """Open the expert's intro receipt inside the office transaction.

    Mirrors ``tasks_db.create_delegated_task`` for a root task (self-stamped
    ``rootTaskId``), but on the transaction client — the shared writer uses
    the global client, whose writes would survive an office rollback.
    """
    row = await tx.delegatedtask.create(
        data={
            "userId": user_id,
            "ownerId": expert_id,
            "createdByType": prisma.enums.TaskCreatedByType.USER,
            "createdById": user_id,
            "title": entry.intro_task_title[:TASK_TITLE_MAX_LENGTH],
            "spec": entry.intro_task_spec[:TASK_SPEC_MAX_LENGTH],
            "status": prisma.enums.DelegatedTaskStatus.QUEUED,
            "ancestorExpertIds": [expert_id],
        }
    )
    stamped = await tx.delegatedtask.update(
        where={"id": row.id}, data={"rootTaskId": row.id}
    )
    return stamped or row


async def _finish_hire(
    user_id: str,
    expert_row: prisma.models.Expert,
    state: str,
    template: prisma.models.Expert,
) -> None:
    """Post-commit follow-ups, all best-effort like the single-hire flow:
    a fresh hire installs the template's preloads, a revived hire resumes
    its paused schedules."""
    if state == "created":
        failed = await experts_db._install_preloads(
            expert_row.id, user_id, template.Workflows or []
        )
        if failed:
            logger.warning(
                f"Office hire: {len(failed)} preload(s) failed for expert "
                f"#{expert_row.id}"
            )
        return
    if state == "revived":
        try:
            await scheduling.resume_expert_schedules(user_id, expert_row.id)
        except Exception:
            logger.exception(
                f"Office hire: failed to resume schedules for expert "
                f"#{expert_row.id}"
            )


async def _create_office_schedule(user_id: str, expert_id: str, cron: str) -> bool:
    """Attach the pack's cadence to the expert's first schedulable workflow.

    Mirrors preload semantics: the cron is recorded on the row first, so a
    scheduler failure leaves the workflow surfaced as "needs setup" instead
    of silently dropping the pack's intent. False when the expert has no
    installed workflow to schedule.
    """
    rows = await prisma.models.ExpertWorkflow.prisma().find_many(
        where={"expertId": expert_id},
        include={"LibraryAgent": True, "StoreListingVersion": True},
        order={"createdAt": "asc"},
    )
    row = next(
        (r for r in rows if r.LibraryAgent is not None and r.scheduleId is None),
        None,
    )
    if row is None or row.LibraryAgent is None:
        return False

    await prisma.models.ExpertWorkflow.prisma().update(
        where={"id": row.id}, data={"scheduleCron": cron}
    )
    user = await get_user_by_id(user_id)
    return await scheduling.create_workflow_schedule(
        workflow_row_id=row.id,
        expert_id=expert_id,
        user_id=user_id,
        cron=cron,
        graph_id=row.LibraryAgent.agentGraphId,
        graph_version=row.LibraryAgent.agentGraphVersion,
        name=(
            row.StoreListingVersion.name
            if row.StoreListingVersion
            else "Expert workflow"
        ),
        user_timezone=get_user_timezone_or_utc(user.timezone if user else None),
    )


def _parse_config(row: prisma.models.OfficeTemplate) -> OfficeConfig | None:
    try:
        return OfficeConfig.model_validate(row.config)
    except ValueError:
        logger.warning(f"Office template '{row.name}' has an invalid config")
        return None


async def _load_templates(
    config: OfficeConfig,
) -> dict[str, prisma.models.Expert] | None:
    """The live template rows for every expert line, or None when any is
    missing/archived — an office must hire completely or not at all."""
    ids = [entry.template_id for entry in config.experts]
    rows = await prisma.models.Expert.prisma().find_many(
        where={"id": {"in": ids}, "isTemplate": True, "isArchived": False},
        include={"Workflows": {"include": {"StoreListingVersion": True}}},
    )
    templates = {row.id: row for row in rows}
    if any(template_id not in templates for template_id in ids):
        return None
    return templates


async def _hydrated_expert(expert_id: str) -> Expert:
    row = await prisma.models.Expert.prisma().find_unique(
        where={"id": expert_id},
        include={
            "Workflows": {
                "include": {"LibraryAgent": True, "StoreListingVersion": True}
            }
        },
    )
    if row is None:
        raise ExpertNotFoundError(expert_id)
    return experts_db._to_model(row)
