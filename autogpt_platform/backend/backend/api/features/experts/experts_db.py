import logging

import prisma.errors
import prisma.models

from backend.api.features.experts.models import Expert, ExpertWorkflowRef, HireResult
from backend.api.features.library import db as library_db

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
    )


def _to_model(row: prisma.models.Expert) -> Expert:
    return Expert(
        id=row.id,
        name=row.name,
        avatar_url=row.avatarUrl,
        role=row.role,
        tagline=row.tagline,
        identity=row.identity,
        is_template=row.isTemplate,
        source_template_id=row.sourceTemplateId,
        is_archived=row.isArchived,
        workflows=[_to_workflow_ref(w) for w in row.Workflows or []],
    )


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
    return [_to_model(row) for row in rows]


async def get_expert(user_id: str, expert_id: str) -> Expert | None:
    row = await prisma.models.Expert.prisma().find_first(
        where={"id": expert_id, "ownerUserId": user_id, "isTemplate": False},
        include=_WORKFLOW_INCLUDE,
    )
    return _to_model(row) if row is not None else None


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
        "identity": template.identity,
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
    return HireResult(expert=_to_model(row), failed_preloads=[])


async def _install_preloads(
    expert_id: str, user_id: str, preloads: list[prisma.models.ExpertWorkflow]
) -> list[str]:
    """Install template preloads into the hiring user's library.

    Honest partial hire: a failed preload is logged and reported, never
    fatal to the hire itself.
    """
    failed: list[str] = []
    for preload in preloads:
        if preload.storeListingVersionId is None:
            continue
        try:
            library_agent = await library_db.add_store_agent_to_library(
                preload.storeListingVersionId, user_id
            )
            await prisma.models.ExpertWorkflow.prisma().create(
                data={
                    "expertId": expert_id,
                    "storeListingVersionId": preload.storeListingVersionId,
                    "libraryAgentId": library_agent.id,
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


async def archive_expert(user_id: str, expert_id: str) -> None:
    updated = await prisma.models.Expert.prisma().update_many(
        where={"id": expert_id, "ownerUserId": user_id, "isTemplate": False},
        data={"isArchived": True},
    )
    if updated == 0:
        raise ExpertNotFoundError(expert_id)
