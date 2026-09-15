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

import logging
from typing import Literal

import prisma.enums
import prisma.models
from pydantic import BaseModel, Field

from backend.api.features.experts.errors import ACTIVE_EXPERT_LIMIT
from backend.api.features.experts.experts_db import count_active_experts
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedWorkflow,
)

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
