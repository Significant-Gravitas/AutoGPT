"""Validate and install raise-flow workflow/skill attachments."""

import logging

import prisma.errors
import prisma.models
import prisma.types
from pydantic import BaseModel

from backend.api.features.experts.models import (
    RaiseAttachment,
    RaiseAttachmentFailure,
    RaiseAttachmentFailureReason,
    RaiseAttachmentKind,
    RaiseAttachmentSource,
)
from backend.api.features.library import db as library_db
from backend.api.features.store import skill_db
from backend.copilot.tools.skills import (
    get_default_skill_with_body,
    read_user_skill_with_body,
)
from backend.data.db import transaction
from backend.util.exceptions import ExpertNotFoundError, NotFoundError

logger = logging.getLogger(__name__)

_WORKFLOW_ROW_INCLUDE: prisma.types.ExpertWorkflowInclude = {
    "LibraryAgent": True,
    "StoreListingVersion": True,
}


class RaiseAttachmentUnavailableError(Exception):
    def __init__(
        self,
        kind: RaiseAttachmentKind,
        source: RaiseAttachmentSource,
        attachment_id: str,
    ):
        super().__init__(f"{source} {kind} {attachment_id} not found or unavailable")
        self.kind = kind
        self.source = source
        self.id = attachment_id


class ResolvedWorkflow(BaseModel):
    attachment: RaiseAttachment
    store_listing_version_id: str | None
    library_agent_id: str | None


class ResolvedSkill(BaseModel):
    """One skill to give the expert.

    ``name`` is what goes on ``Expert.skills``. ``marketplace_slug`` is set for
    a Hub listing, whose SKILL.md is installed into the expert's folder; a
    library skill is copied from the user's own folder instead.
    """

    attachment: RaiseAttachment
    name: str
    marketplace_slug: str | None = None


class ResolvedRaiseAttachments(BaseModel):
    workflows: list[ResolvedWorkflow]
    skills: list[ResolvedSkill]

    @property
    def skill_names(self) -> list[str]:
        return [skill.name for skill in self.skills]

    @property
    def library_skill_names(self) -> list[str]:
        return [s.name for s in self.skills if s.marketplace_slug is None]


async def resolve_attachments(
    user_id: str, attachments: list[RaiseAttachment]
) -> ResolvedRaiseAttachments:
    """Pre-check every attachment. Raises if any item is not attachable."""
    workflows: list[ResolvedWorkflow] = []
    skills: list[ResolvedSkill] = []
    seen: set[tuple[str, str, str]] = set()
    for attachment in attachments:
        key = _dedupe_key(attachment)
        if key in seen:
            continue
        seen.add(key)
        if attachment.kind == "workflow":
            workflows.append(await _resolve_workflow(user_id, attachment))
            continue
        skills.append(await _resolve_skill(user_id, attachment))
    return ResolvedRaiseAttachments(workflows=workflows, skills=skills)


async def install_marketplace_skills(
    user_id: str,
    expert_id: str,
    skills: list[ResolvedSkill],
) -> list[RaiseAttachmentFailure]:
    """Copy each Hub listing's SKILL.md into the expert's own skill folder.

    Library skills are the caller's job — they are copied from the user's own
    folder. Failures are reported, not fatal, as workflow installs are.
    """
    failed: list[RaiseAttachmentFailure] = []
    for resolved in skills:
        if resolved.marketplace_slug is None:
            continue
        try:
            await skill_db.install_marketplace_skill(
                user_id, resolved.marketplace_slug, expert_id=expert_id
            )
        except NotFoundError:
            failed.append(_failure(resolved.attachment, "unavailable"))
            logger.warning(
                f"Marketplace skill {resolved.marketplace_slug!r} became "
                f"unavailable while raising expert #{expert_id} for user #{user_id}"
            )
        except Exception:
            failed.append(_failure(resolved.attachment, "installation_failed"))
            logger.exception(
                f"Failed to install marketplace skill "
                f"{resolved.marketplace_slug!r} on raised expert #{expert_id} "
                f"for user #{user_id}"
            )
    return failed


async def install_workflows(
    user_id: str,
    expert_id: str,
    workflows: list[ResolvedWorkflow],
) -> list[RaiseAttachmentFailure]:
    """Install validated workflows. Failures are reported, not fatal."""
    failed: list[RaiseAttachmentFailure] = []
    for resolved in workflows:
        try:
            await _install_resolved_workflow(user_id, expert_id, resolved)
        except (RaiseAttachmentUnavailableError, NotFoundError):
            failed.append(_failure(resolved.attachment, "unavailable"))
            logger.warning(
                f"{resolved.attachment.source} workflow {resolved.attachment.id} "
                f"became unavailable while raising expert #{expert_id} "
                f"for user #{user_id}"
            )
        except Exception:
            failed.append(_failure(resolved.attachment, "installation_failed"))
            logger.exception(
                f"Failed to install {resolved.attachment.source} workflow "
                f"{resolved.attachment.id} on raised expert #{expert_id} "
                f"for user #{user_id}"
            )
    return failed


async def install_marketplace_workflow(
    user_id: str,
    expert_id: str,
    store_listing_version_id: str,
) -> None:
    """Install a marketplace listing and attach it to the expert.

    A concurrent or retried raise can win the ExpertWorkflow insert; that row
    is the same attachment, so a unique violation counts as success. The
    re-check runs after the failed transaction has rolled back, never inside
    it — Postgres aborts a transaction on the first failed statement.
    """
    try:
        await _install_marketplace_workflow_once(
            user_id, expert_id, store_listing_version_id
        )
    except prisma.errors.UniqueViolationError:
        raced = await _existing_listing_workflow(expert_id, store_listing_version_id)
        if raced is None:
            raise


async def _install_marketplace_workflow_once(
    user_id: str,
    expert_id: str,
    store_listing_version_id: str,
) -> None:
    async with transaction() as tx:
        is_installable = (
            await library_db.is_store_listing_version_available_for_install(
                store_listing_version_id,
                tx=tx,
                lock_rows=True,
            )
        )
        if not is_installable:
            raise RaiseAttachmentUnavailableError(
                "workflow", "marketplace", store_listing_version_id
            )
        expert = await tx.expert.find_first(
            where={
                "id": expert_id,
                "ownerUserId": user_id,
                "isTemplate": False,
                "isArchived": False,
            }
        )
        if expert is None:
            raise ExpertNotFoundError(expert_id)
        library_agent = await library_db.add_store_agent_to_library_in_transaction(
            store_listing_version_id, user_id, tx
        )
        await tx.expertworkflow.create(
            data={
                "expertId": expert_id,
                "storeListingVersionId": store_listing_version_id,
                "libraryAgentId": library_agent.id,
            }
        )


async def _resolve_workflow(
    user_id: str, attachment: RaiseAttachment
) -> ResolvedWorkflow:
    if attachment.source == "marketplace":
        await _validate_marketplace_listing(attachment.id)
        return ResolvedWorkflow(
            attachment=attachment,
            store_listing_version_id=attachment.id,
            library_agent_id=None,
        )
    row = await _library_agent_row(user_id, attachment.id)
    return ResolvedWorkflow(
        attachment=attachment,
        store_listing_version_id=None,
        library_agent_id=row.id,
    )


async def _resolve_skill(user_id: str, attachment: RaiseAttachment) -> ResolvedSkill:
    slug = attachment.id.strip().lower()
    if attachment.source == "marketplace":
        try:
            await skill_db.get_marketplace_skill(slug)
        except NotFoundError:
            raise RaiseAttachmentUnavailableError("skill", "marketplace", slug)
        # The install names the copy after the listing slug, so that is the
        # name the expert's row has to carry for the two to line up.
        return ResolvedSkill(attachment=attachment, name=slug, marketplace_slug=slug)
    default = get_default_skill_with_body(slug)
    if default is not None:
        return ResolvedSkill(attachment=attachment, name=default.name)
    stored = await read_user_skill_with_body(user_id, slug)
    if stored is None:
        raise RaiseAttachmentUnavailableError("skill", "library", slug)
    return ResolvedSkill(attachment=attachment, name=stored.name)


async def _validate_marketplace_listing(store_listing_version_id: str) -> None:
    is_installable = await library_db.is_store_listing_version_available_for_install(
        store_listing_version_id
    )
    if not is_installable:
        raise RaiseAttachmentUnavailableError(
            "workflow", "marketplace", store_listing_version_id
        )


async def _install_resolved_workflow(
    user_id: str, expert_id: str, resolved: ResolvedWorkflow
) -> None:
    if resolved.library_agent_id is not None:
        await _link_library_workflow(expert_id, resolved.library_agent_id)
        return
    if resolved.store_listing_version_id is None:
        raise RaiseAttachmentUnavailableError(
            "workflow", resolved.attachment.source, resolved.attachment.id
        )
    await install_marketplace_workflow(
        user_id, expert_id, resolved.store_listing_version_id
    )


async def _link_library_workflow(expert_id: str, library_agent_id: str) -> None:
    existing = await _existing_library_workflow(expert_id, library_agent_id)
    if existing is not None:
        return
    try:
        await prisma.models.ExpertWorkflow.prisma().create(
            data={"expertId": expert_id, "libraryAgentId": library_agent_id}
        )
    except prisma.errors.UniqueViolationError:
        raced = await _existing_library_workflow(expert_id, library_agent_id)
        if raced is None:
            raise


async def _library_agent_row(
    user_id: str, library_agent_id: str
) -> prisma.models.LibraryAgent:
    row = await prisma.models.LibraryAgent.prisma().find_first(
        where={
            "id": library_agent_id,
            "userId": user_id,
            "isDeleted": False,
        }
    )
    if row is None:
        raise RaiseAttachmentUnavailableError("workflow", "library", library_agent_id)
    return row


async def _existing_library_workflow(
    expert_id: str,
    library_agent_id: str,
) -> prisma.models.ExpertWorkflow | None:
    return await prisma.models.ExpertWorkflow.prisma().find_first(
        where={"expertId": expert_id, "libraryAgentId": library_agent_id},
        include=_WORKFLOW_ROW_INCLUDE,
    )


async def _existing_listing_workflow(
    expert_id: str,
    store_listing_version_id: str,
) -> prisma.models.ExpertWorkflow | None:
    return await prisma.models.ExpertWorkflow.prisma().find_first(
        where={
            "expertId": expert_id,
            "storeListingVersionId": store_listing_version_id,
        },
        include=_WORKFLOW_ROW_INCLUDE,
    )


def _dedupe_key(attachment: RaiseAttachment) -> tuple[str, str, str]:
    attachment_id = attachment.id
    # Both skill sources are slugs, which are case-insensitive.
    if attachment.kind == "skill":
        attachment_id = attachment.id.strip().lower()
    return (attachment.kind, attachment.source, attachment_id)


def _failure(
    attachment: RaiseAttachment, reason: RaiseAttachmentFailureReason
) -> RaiseAttachmentFailure:
    return RaiseAttachmentFailure(
        kind=attachment.kind,
        source=attachment.source,
        id=attachment.id,
        reason=reason,
    )
