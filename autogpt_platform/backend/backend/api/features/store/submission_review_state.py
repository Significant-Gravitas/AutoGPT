"""Locked workflow submission state and public fallback selection."""

import prisma.enums
import prisma.models
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.util.exceptions import ConflictError

from .store_listing_versions import installable_store_version_where


class LockedReviewTarget(BaseModel):
    active_version_id: str | None
    is_deleted: bool
    submission_status: prisma.enums.SubmissionStatus


async def lock_review_target(tx: prisma.Prisma, version_id: str) -> LockedReviewTarget:
    """Lock the listing before its version, then read the current review state."""
    rows = await query_raw_with_schema(
        'SELECT l."activeVersionId" AS active_version_id, v."isDeleted" AS is_deleted, '
        'v."submissionStatus"::text AS submission_status '
        'FROM {schema_prefix}"StoreListing" l JOIN {schema_prefix}"StoreListingVersion" v '
        'ON v."storeListingId" = l.id WHERE v.id = $1 FOR UPDATE OF l, v',
        version_id,
        model=LockedReviewTarget,
        client=tx,
    )
    if not rows:
        raise ConflictError(
            "This submission was removed during review. Refresh the listing."
        )
    return rows[0]


async def withdraw_store_version(
    tx: prisma.Prisma, version: prisma.models.StoreListingVersion
) -> str | None:
    """Adjust the listing only if the locked version is still approved.

    The caller writes the rejection in the same transaction. Return the
    reactivated version ID, if any. Clear the listing's flag only when no
    other undeleted approval remains.
    """
    target = await lock_review_target(tx, version.id)
    if target.submission_status != prisma.enums.SubmissionStatus.APPROVED:
        return None

    listings = prisma.models.StoreListing.prisma(tx)
    if target.active_version_id == version.id:
        fallback = await prisma.models.StoreListingVersion.prisma(tx).find_first(
            where={
                **installable_store_version_where(),
                "storeListingId": version.storeListingId,
                "id": {"not": version.id},
            },
            order={"version": "desc"},
        )
        await listings.update(
            where={"id": version.storeListingId},
            data={
                "ActiveVersion": (
                    {"connect": {"id": fallback.id}}
                    if fallback
                    else {"disconnect": True}
                )
            },
        )
        if fallback:
            return fallback.id

    # Hidden sub-workflows still count as approved without becoming public.
    await listings.update_many(
        where={
            "id": version.storeListingId,
            "hasApprovedVersion": True,
            "Versions": {
                "none": {
                    "id": {"not": version.id},
                    "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                    "isDeleted": False,
                }
            },
        },
        data={"hasApprovedVersion": False},
    )
    return None
