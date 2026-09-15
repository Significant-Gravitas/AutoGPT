"""Reading an uploaded expert package, before anything is created.

Importing an expert is two requests, not one: this half answers "what is in this
file, and what would happen if I imported it" and writes nothing at all. That is
what lets the review dialog show the expert, let the user take skills and
workflows out, and only then create.

The interesting question is each workflow. A package carries both a marketplace
reference and the graph itself, and which one an import should use depends on
this installation: a published listing the importer can see is worth far more
than a copy, because it keeps them on the real agent with its updates and its
creator's attribution. Resolving that here means the dialog can say which of the
two each workflow will land as, before the user commits to anything.
"""

import io
import logging
from typing import Literal

import prisma.enums
import prisma.models
from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import Headers

from backend.api.features.experts import experts_db, scheduling
from backend.api.features.experts.errors import ACTIVE_EXPERT_LIMIT
from backend.api.features.experts.experts_db import count_active_experts
from backend.api.features.experts.models import EXPERT_NAME_MAX_LENGTH, Expert
from backend.api.features.experts.package_model import (
    MAX_PACKAGE_SKILLS,
    MAX_PACKAGE_WORKFLOWS,
    ExpertManifest,
    ExpertPackage,
    ExpertPackageError,
    PackagedSkill,
    PackagedWorkflow,
)
from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.features.store.media import upload_media
from backend.copilot.tools.skills import parse_skill_markdown, store_user_skill
from backend.data import graph as graph_db
from backend.data.user import get_user_by_id
from backend.integrations.webhooks.graph_lifecycle_hooks import before_graph_activate
from backend.util.timezone_utils import get_user_timezone_or_utc

logger = logging.getLogger(__name__)

WorkflowSource = Literal["store", "graph", "unresolvable"]
AvatarKind = Literal["file", "url", "none"]


class ExpertPackageIssue(BaseModel):
    """Something the user should know before importing. A ``warning`` still
    imports; an ``error`` is why the dialog's confirm button is disabled."""

    code: str
    message: str
    path: str | None = None


class WorkflowResolution(BaseModel):
    """Which of a workflow's two sources this installation would actually use.

    ``index`` is the workflow's position in the manifest, which is how the
    import request names the ones the user removed.
    """

    index: int
    name: str
    source: WorkflowSource
    store_listing_version_id: str | None = None
    store_listing_slug: str | None = None
    schedule_cron: str | None = None
    reason: str | None = None


class PackagedFileInfo(BaseModel):
    path: str
    size_bytes: int


class PackagedSkillInfo(BaseModel):
    """A skill as the review dialog lists it: the card, plus the files that
    would be written, so "3 files" is answerable without unzipping again."""

    slug: str
    name: str
    description: str
    files: list[PackagedFileInfo] = Field(default_factory=list)


class ExpertPackagePreview(BaseModel):
    """What an uploaded file turns out to hold, and what importing it would do."""

    manifest: ExpertManifest
    avatar_kind: AvatarKind
    skills: list[PackagedSkillInfo] = Field(default_factory=list)
    workflows: list[WorkflowResolution] = Field(default_factory=list)
    warnings: list[ExpertPackageIssue] = Field(default_factory=list)
    errors: list[ExpertPackageIssue] = Field(default_factory=list)


async def preview_package(user_id: str, package: ExpertPackage) -> ExpertPackagePreview:
    """Describe *package* for the review dialog. Writes nothing."""
    workflows = [
        await resolve_workflow(workflow, index)
        for index, workflow in enumerate(package.manifest.workflows)
    ]
    return ExpertPackagePreview(
        manifest=package.manifest,
        avatar_kind=package.manifest.avatar.kind if package.manifest.avatar else "none",
        skills=[
            PackagedSkillInfo(
                slug=card.slug,
                name=card.name,
                description=card.description,
                files=_files(package, card.slug),
            )
            for card in package.manifest.skills
        ],
        workflows=workflows,
        warnings=_warnings(package.manifest.workflows, workflows),
        errors=await _errors(user_id),
    )


def _warnings(
    packaged: list[PackagedWorkflow], resolved: list[WorkflowResolution]
) -> list[ExpertPackageIssue]:
    """What the user should see before confirming. Neither of these blocks the
    import: the rest of the expert still arrives."""
    warnings = []
    for workflow, resolution in zip(packaged, resolved):
        if resolution.source == "unresolvable":
            warnings.append(
                ExpertPackageIssue(
                    code="unresolvable_workflow",
                    message=(
                        f"'{resolution.name}' cannot be imported: "
                        f"{resolution.reason}. The rest of the expert still "
                        "imports."
                    ),
                )
            )
        elif resolution.source == "graph" and (
            workflow.store_listing_version_id or workflow.store_listing_slug
        ):
            # A silent downgrade would leave the user wondering why their
            # marketplace agent stopped getting its creator's updates.
            warnings.append(
                ExpertPackageIssue(
                    code="workflow_not_in_marketplace",
                    message=(
                        f"'{resolution.name}' is not on this marketplace; a "
                        "private copy will be created instead."
                    ),
                )
            )
    return warnings


async def resolve_workflow(
    workflow: PackagedWorkflow, index: int
) -> WorkflowResolution:
    """The marketplace listing this installation can see, else the graph in the
    file, else nothing importable."""
    resolution = WorkflowResolution(
        index=index,
        name=workflow.name,
        source="graph" if workflow.graph else "unresolvable",
        schedule_cron=workflow.schedule_cron,
    )
    if version_id := await _listing_version_id(workflow):
        return resolution.model_copy(
            update={
                "source": "store",
                "store_listing_version_id": version_id,
                "store_listing_slug": workflow.store_listing_slug,
            }
        )
    if resolution.source == "unresolvable":
        resolution.reason = (
            "its agent is not on this marketplace and the file carries no copy"
        )
    return resolution


async def _listing_version_id(workflow: PackagedWorkflow) -> str | None:
    """The version the reference points at, if it is on this marketplace and
    still published."""
    if workflow.store_listing_version_id:
        version = await prisma.models.StoreListingVersion.prisma().find_first(
            where={
                "id": workflow.store_listing_version_id,
                "isDeleted": False,
                "isAvailable": True,
                "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
            }
        )
        if version:
            return version.id
    if not (workflow.store_listing_slug and workflow.creator_username):
        # A slug alone is ambiguous: (owner, slug) is what a listing is unique
        # on, so without the creator this could resolve to somebody else's.
        return None
    # The username is unique, so this match is deterministic: at most one
    # listing, exactly as ``seed._resolve_active_version_id`` relies on.
    listing = await prisma.models.StoreListing.prisma().find_first(
        where={
            "slug": workflow.store_listing_slug,
            "isDeleted": False,
            "hasApprovedVersion": True,
            "CreatorProfile": {"is": {"username": workflow.creator_username}},
        }
    )
    return listing.activeVersionId if listing else None


def _files(package: ExpertPackage, slug: str) -> list[PackagedFileInfo]:
    skill = package.skills.get(slug)
    if skill is None:
        return []
    return [
        PackagedFileInfo(path="SKILL.md", size_bytes=len(skill.skill_md.encode())),
        *(
            PackagedFileInfo(path=f.relative_path, size_bytes=f.size_bytes)
            for f in skill.files
        ),
    ]


async def _errors(user_id: str) -> list[ExpertPackageIssue]:
    """Why importing would fail outright. Read without a lock — the creating
    transaction re-enforces the cap — because a preview must not hold one."""
    if await count_active_experts(user_id) < ACTIVE_EXPERT_LIMIT:
        return []
    return [
        ExpertPackageIssue(
            code="active_expert_limit",
            message=(
                f"You already have {ACTIVE_EXPERT_LIMIT} active experts. "
                "Let one go before importing another."
            ),
        )
    ]


class ExpertImportWorkflowEdit(BaseModel):
    """What the review dialog lets a user change about one workflow: whether
    its schedule should start. The cadence itself comes from the file."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    schedule_enabled: bool = False


class ExpertImportEdits(BaseModel):
    """The review dialog's whole output: a rename, and what to leave out.

    Deliberately small. Everything else about the expert is what the file says,
    so an import stays a faithful restore rather than a second authoring tool.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=EXPERT_NAME_MAX_LENGTH)
    removed_skill_slugs: list[str] = Field(
        default_factory=list, max_length=MAX_PACKAGE_SKILLS
    )
    removed_workflow_indices: list[int] = Field(
        default_factory=list, max_length=MAX_PACKAGE_WORKFLOWS
    )
    workflows: list[ExpertImportWorkflowEdit] = Field(
        default_factory=list, max_length=MAX_PACKAGE_WORKFLOWS
    )


class ExpertImportResult(BaseModel):
    """An honest partial import: the expert exists, and these parts of it did
    not make it. Mirrors ``HireResult.failed_preloads``."""

    expert: Expert
    failed_workflows: list[str] = Field(default_factory=list)
    failed_skills: list[str] = Field(default_factory=list)
    warnings: list[ExpertPackageIssue] = Field(default_factory=list)


async def import_package(
    user_id: str, package: ExpertPackage, edits: ExpertImportEdits
) -> ExpertImportResult:
    """Create *user_id*'s own expert from *package*.

    Partial by design: a skill or a workflow that cannot be installed is
    reported rather than rolled back, because an expert missing one of its
    agents is far more use than no expert at all.
    """
    manifest = package.manifest
    removed = set(edits.removed_skill_slugs)
    skills = [card for card in manifest.skills if card.slug not in removed]
    workflows = [
        (index, workflow)
        for index, workflow in enumerate(manifest.workflows)
        if index not in set(edits.removed_workflow_indices)
    ]
    avatar_url, warnings = await _avatar(user_id, package)
    row = await experts_db.create_imported_expert(
        user_id,
        name=edits.name or manifest.identity.name,
        role=manifest.identity.role,
        tagline=manifest.identity.tagline,
        bio=manifest.identity.bio,
        color=manifest.identity.color,
        categories=manifest.identity.categories,
        identity=manifest.soul.identity,
        voice_preferences=manifest.soul.voice_preferences,
        boundaries=manifest.soul.boundaries,
        avatar_url=avatar_url,
        day_one=manifest.day_one,
        tool_profile=manifest.tool_profile,
    )
    failed_skills = await install_package_skills(user_id, row.id, package, skills)
    failed_workflows = await _install_workflows(user_id, row.id, workflows, edits)
    expert = await experts_db.get_expert(user_id, row.id)
    if expert is None:
        raise ExpertPackageError("the imported expert could not be read back")
    return ExpertImportResult(
        expert=expert,
        failed_workflows=failed_workflows,
        failed_skills=failed_skills,
        warnings=warnings,
    )


async def install_package_skills(
    user_id: str,
    expert_id: str,
    package: ExpertPackage,
    skills: list[PackagedSkill],
) -> list[str]:
    """Write each skill's whole folder into the expert's own workspace.

    Shared with hiring a published template, which installs the same packaged
    skills from the same zip. ``store_user_skill`` records the name on the
    expert row itself, so a skill that fails leaves no name behind and the
    expert never lists one it does not have.
    """
    failed: list[str] = []
    for card in skills:
        stored = package.skills.get(card.slug)
        parsed = parse_skill_markdown(stored.skill_md) if stored else None
        if stored is None or parsed is None:
            failed.append(card.name)
            continue
        try:
            await store_user_skill(
                user_id,
                name=parsed.name or card.name,
                description=parsed.description,
                body=parsed.body,
                triggers=list(parsed.triggers),
                files=stored.files,
                expert_id=expert_id,
            )
        except Exception:
            logger.exception(
                f"Failed to install packaged skill {card.slug!r} on expert #{expert_id}"
            )
            failed.append(card.name)
    return failed


async def _install_workflows(
    user_id: str,
    expert_id: str,
    workflows: list[tuple[int, PackagedWorkflow]],
    edits: ExpertImportEdits,
) -> list[str]:
    """Rows first, schedules second — the ordering ``_install_preloads``
    documents: creating a schedule resolves credentials scoped to the expert,
    which seeds its allow-list from the workflows installed so far, so
    interleaving would freeze that list after the first one."""
    enabled = {e.index for e in edits.workflows if e.schedule_enabled}
    failed: list[str] = []
    installed: list[tuple[str, PackagedWorkflow, library_model.LibraryAgent]] = []
    for index, workflow in workflows:
        try:
            agent = await _library_agent(user_id, workflow, index)
            row = await prisma.models.ExpertWorkflow.prisma().create(
                data={
                    "expertId": expert_id,
                    "storeListingVersionId": await _listing_version_id(workflow),
                    "libraryAgentId": agent.id,
                    "scheduleCron": workflow.schedule_cron,
                }
            )
        except Exception:
            logger.exception(
                f"Failed to install workflow {index} on expert #{expert_id}"
            )
            failed.append(workflow.name or f"Workflow {index + 1}")
            continue
        if index in enabled and workflow.schedule_cron:
            installed.append((row.id, workflow, agent))
    if not installed:
        return failed
    user = await get_user_by_id(user_id)
    timezone = get_user_timezone_or_utc(user.timezone if user else None)
    for row_id, workflow, agent in installed:
        await scheduling.create_workflow_schedule(
            workflow_row_id=row_id,
            expert_id=expert_id,
            user_id=user_id,
            cron=workflow.schedule_cron or "",
            graph_id=agent.graph_id,
            graph_version=agent.graph_version,
            name=workflow.name or "Expert workflow",
            user_timezone=timezone,
        )
    return failed


async def _library_agent(
    user_id: str, workflow: PackagedWorkflow, index: int
) -> library_model.LibraryAgent:
    """The importer's own library agent for this workflow: the published one
    when this marketplace has it, else a private copy of the embedded graph."""
    if version_id := await _listing_version_id(workflow):
        return await library_db.add_store_agent_to_library(version_id, user_id)
    if workflow.graph is None:
        raise ValueError(f"workflow {index} has no source to install")
    # The POST /api/v1/graphs sequence: ids are reassigned so the graph
    # becomes this user's, and credentials are validated before anything is
    # persisted so a bad one cannot leave a half-saved graph behind.
    graph = graph_db.make_graph_model(workflow.graph, user_id)
    graph.reassign_ids(user_id=user_id, reassign_graph_id=True)
    graph.validate_graph(for_run=False)
    graph = await before_graph_activate(graph, user_id=user_id)
    await graph_db.create_graph(graph, user_id=user_id)
    # The parent graph is always the first entry; any sub-graph agents follow.
    return (await library_db.create_library_agent(graph, user_id))[0]


async def _avatar(
    user_id: str, package: ExpertPackage
) -> tuple[str | None, list[ExpertPackageIssue]]:
    """The imported expert's picture. Embedded bytes go through the media
    pipeline — magic-byte checks and a virus scan — because a package is
    user-supplied input whoever it came from."""
    avatar = package.manifest.avatar
    if avatar is None:
        return None, []
    if avatar.kind == "url":
        return (avatar.url if (avatar.url or "").startswith("/") else None), []
    if package.avatar_bytes is None:
        return None, []
    upload = UploadFile(
        file=io.BytesIO(package.avatar_bytes),
        filename=avatar.path,
        headers=Headers({"content-type": package.avatar_mime or "image/png"}),
    )
    try:
        return await upload_media(user_id, upload), []
    except Exception as exc:
        logger.warning(f"Imported avatar could not be stored: {exc}")
        return None, [
            ExpertPackageIssue(
                code="avatar_not_imported",
                message="The expert's picture could not be imported.",
                path=avatar.path,
            )
        ]
