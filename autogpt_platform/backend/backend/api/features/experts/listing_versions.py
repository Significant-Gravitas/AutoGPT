"""Which marketplace version a workflow reference is worth, by the library's
own test.

Both the importer and the publisher have to answer the same question about a
store listing version: would a hire that lands on it install anything? The
library already has the one predicate for that — approved, available, not
deleted, on a listing that is not deleted — and both go through it here so a
reference that a hire would refuse is never resolved as usable.
"""

import prisma.models

from backend.api.features.store.store_listing_versions import (
    installable_store_version_where,
)


async def installable_version(
    version_id: str,
) -> prisma.models.StoreListingVersion | None:
    """The version, with the listing and creator an export names, if a
    library install would accept it; ``None`` otherwise."""
    return await prisma.models.StoreListingVersion.prisma().find_first(
        where={"id": version_id, **installable_store_version_where()},
        include={"StoreListing": {"include": {"CreatorProfile": True}}},
    )


async def installable_active_version(
    listing: prisma.models.StoreListing | None,
) -> prisma.models.StoreListingVersion | None:
    """The listing's active version, held to the same test. A listing whose
    active version is pending, hidden or deleted resolves to nothing rather
    than to a version a hire could not install."""
    if listing is None or listing.isDeleted or not listing.activeVersionId:
        return None
    return await installable_version(listing.activeVersionId)
