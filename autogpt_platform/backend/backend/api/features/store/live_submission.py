"""The version of a creator's listing that the Marketplace shows."""

import prisma.enums
import prisma.models
import prisma.types

from . import model as store_model


async def get_live_submission(
    user_id: str, graph_id: str, organization_id: str | None = None
) -> store_model.LiveStoreSubmission | None:
    """The live version of the caller's listing for ``graph_id``, with its
    recommended schedule, or None when the graph has no listing of theirs or
    nothing of it is live.

    The live version is ``StoreListing.activeVersionId``, read with the
    conditions the ``StoreAgent`` view puts on it. It isn't simply the newest
    approved version: a sub-workflow's approved version is hidden, and a
    rejection moves the listing back to an older one.

    Ownership follows ``get_store_submissions``: with ``organization_id``
    (membership-verified RequestContext), the org's listings and the caller's
    tenant-less ones; without it, the caller's own.
    """
    where: prisma.types.StoreListingWhereInput = {
        "agentGraphId": graph_id,
        "isDeleted": False,
        "hasApprovedVersion": True,
    }
    if organization_id is not None:
        where["OR"] = [
            {"owningOrgId": organization_id},
            {"owningUserId": user_id, "owningOrgId": None},
        ]
    else:
        where["owningUserId"] = user_id

    # The StoreSubmission view has no schedule column, so it comes from the
    # live version's own row.
    listing = await prisma.models.StoreListing.prisma().find_first(
        where=where, include={"ActiveVersion": True}
    )
    if not listing or not (active := listing.ActiveVersion):
        return None

    live = await prisma.models.StoreSubmission.prisma().find_first(
        where={
            "listing_version_id": active.id,
            "status": prisma.enums.SubmissionStatus.APPROVED,
        }
    )
    if not live:
        return None
    return store_model.LiveStoreSubmission.from_live_version(live, active)
