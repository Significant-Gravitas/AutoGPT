"""Publishing a library skill as a marketplace listing, and reviewing it.

A submission snapshots the creator's own ``SKILL.md`` at submit time, so
editing the library copy afterwards never changes what installers get — only
a new submission does. The live version stays served throughout: approval is
what promotes a pending version, and a rejection leaves the shelf untouched.
"""

import datetime

import prisma.enums
import prisma.models

from backend.copilot.tools.skills import read_user_skill_with_body
from backend.data.db import transaction
from backend.util.exceptions import NotFoundError, PreconditionFailed

from . import skill_model


async def submit_skill(
    user_id: str, request: skill_model.SkillSubmissionRequest
) -> skill_model.SkillSubmission:
    """Create a pending version for the caller's library skill.

    Re-publishing an already-listed skill adds a version rather than editing
    the live one, so what the marketplace serves only changes on approval.
    """
    slug = request.skill_name.strip().lower()
    skill = await read_user_skill_with_body(user_id, slug)
    if skill is None:
        raise NotFoundError(f"Skill '{slug}' is not in your library")
    await _require_profile(user_id)

    async with transaction() as tx:
        listing = await prisma.models.SkillListing.prisma(tx).find_unique(
            where={"slug": slug}
        )
        if listing is None:
            listing = await prisma.models.SkillListing.prisma(tx).create(
                data={"slug": slug, "owningUserId": user_id}
            )
        elif listing.owningUserId != user_id:
            raise PreconditionFailed(
                f"The marketplace slug '{slug}' is taken. Rename your skill "
                "and publish it again."
            )
        version = await prisma.models.SkillListingVersion.prisma(tx).create(
            data={
                "skillListingId": listing.id,
                "version": await _next_version(tx, listing.id),
                "name": skill.name,
                "description": skill.description,
                "body": skill.body,
                "triggers": list(skill.triggers),
                "categories": request.categories,
                "requiredProviders": request.required_providers,
                "sourceSkillSlug": slug,
                "changesSummary": request.changes_summary or "Initial submission",
                "submissionStatus": prisma.enums.SubmissionStatus.PENDING,
                "submittedAt": datetime.datetime.now(datetime.timezone.utc),
            }
        )
    return skill_model.SkillSubmission.from_db(version, listing)


async def list_my_skill_submissions(
    user_id: str,
) -> list[skill_model.SkillSubmission]:
    listings = await prisma.models.SkillListing.prisma().find_many(
        where={"owningUserId": user_id, "isDeleted": False},
        include={"Versions": True},
        order={"createdAt": "desc"},
    )
    return [
        skill_model.SkillSubmission.from_db(version, listing)
        for listing in listings
        for version in sorted(
            listing.Versions or [], key=lambda v: v.version, reverse=True
        )
        if not version.isDeleted
    ]


async def edit_skill_submission(
    user_id: str,
    skill_listing_version_id: str,
    request: skill_model.SkillSubmissionRequest,
) -> skill_model.SkillSubmission:
    """Update a pending submission and re-snapshot the library skill.

    Only a pending version is editable — an approved one is what the
    marketplace serves, and a rejected one is a closed record.
    """
    version, listing = await _owned_version(user_id, skill_listing_version_id)
    if version.submissionStatus != prisma.enums.SubmissionStatus.PENDING:
        raise PreconditionFailed("Only a pending submission can be edited")
    # The slug and sourceSkillSlug were fixed at submit time, so re-pointing the
    # edit at another skill would serve b's content under a's slug.
    slug = request.skill_name.strip().lower()
    if slug != listing.slug:
        raise PreconditionFailed(
            f"This submission publishes '{listing.slug}'. Publish '{slug}' as "
            "its own listing instead."
        )
    skill = await read_user_skill_with_body(user_id, slug)
    if skill is None:
        raise NotFoundError(f"Skill '{slug}' is not in your library")

    updated = await prisma.models.SkillListingVersion.prisma().update(
        where={"id": version.id},
        data={
            "name": skill.name,
            "description": skill.description,
            "body": skill.body,
            "triggers": list(skill.triggers),
            "categories": request.categories,
            "requiredProviders": request.required_providers,
            "changesSummary": request.changes_summary or version.changesSummary,
        },
    )
    if updated is None:
        raise NotFoundError(f"Submission #{skill_listing_version_id} not found")
    return skill_model.SkillSubmission.from_db(updated, listing)


async def review_skill_submission(
    skill_listing_version_id: str,
    *,
    is_approved: bool,
    reviewer_id: str,
    comments: str,
    internal_comments: str = "",
) -> skill_model.SkillSubmission:
    """Approve or reject a pending submission.

    Approval promotes this version to the one the marketplace serves; a
    rejection records the verdict and leaves the live version alone.
    """
    status = (
        prisma.enums.SubmissionStatus.APPROVED
        if is_approved
        else prisma.enums.SubmissionStatus.REJECTED
    )
    async with transaction() as tx:
        # PENDING sits in the WHERE so a second verdict cannot overwrite the
        # first; without it, re-approving a rejected version re-promotes it.
        reviewed = await prisma.models.SkillListingVersion.prisma(tx).update_many(
            where={
                "id": skill_listing_version_id,
                "submissionStatus": prisma.enums.SubmissionStatus.PENDING,
            },
            data={
                "submissionStatus": status,
                "reviewerId": reviewer_id,
                "reviewComments": comments,
                "internalComments": internal_comments,
            },
        )
        updated = await prisma.models.SkillListingVersion.prisma(tx).find_unique(
            where={"id": skill_listing_version_id}, include={"SkillListing": True}
        )
        if updated is None or updated.SkillListing is None:
            raise NotFoundError(f"Submission #{skill_listing_version_id} not found")
        if reviewed == 0:
            raise PreconditionFailed("Only a pending submission can be reviewed")
        listing = updated.SkillListing
        if is_approved:
            listing = (
                await prisma.models.SkillListing.prisma(tx).update(
                    where={"id": listing.id},
                    data={"activeVersionId": updated.id, "hasApprovedVersion": True},
                )
                or listing
            )
    return skill_model.SkillSubmission.from_db(updated, listing)


async def list_pending_skill_submissions() -> list[skill_model.SkillSubmission]:
    versions = await prisma.models.SkillListingVersion.prisma().find_many(
        where={
            "submissionStatus": prisma.enums.SubmissionStatus.PENDING,
            "isDeleted": False,
            "SkillListing": {"is": {"isDeleted": False}},
        },
        include={"SkillListing": True},
        order={"createdAt": "asc"},
    )
    return [
        skill_model.SkillSubmission.from_db(v, v.SkillListing)
        for v in versions
        if v.SkillListing is not None
    ]


async def _owned_version(
    user_id: str, skill_listing_version_id: str
) -> tuple[prisma.models.SkillListingVersion, prisma.models.SkillListing]:
    version = await prisma.models.SkillListingVersion.prisma().find_unique(
        where={"id": skill_listing_version_id}, include={"SkillListing": True}
    )
    if (
        version is None
        or version.SkillListing is None
        or version.SkillListing.owningUserId != user_id
    ):
        raise NotFoundError(f"Submission #{skill_listing_version_id} not found")
    return version, version.SkillListing


async def _next_version(tx, skill_listing_id: str) -> int:
    latest = await prisma.models.SkillListingVersion.prisma(tx).find_first(
        where={"skillListingId": skill_listing_id}, order={"version": "desc"}
    )
    return (latest.version + 1) if latest else 1


async def _require_profile(user_id: str) -> None:
    profile = await prisma.models.Profile.prisma().find_unique(
        where={"userId": user_id}
    )
    if profile is None:
        raise PreconditionFailed(
            "User must create a Marketplace Profile before publishing a skill"
        )
