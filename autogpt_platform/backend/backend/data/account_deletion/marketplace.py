"""The account's marketplace presence: its reviews, and the listings that do
not go to the "deleted user" account (`shared_agents.py`).

Both reference the account with ON DELETE RESTRICT, so they must go before
the account's rows can. Listing versions take their submissions and the
reviews other accounts left on them along; their search embeddings carry no
foreign key and are deleted here. Their media goes with the account's files
(`storage.py`), which keeps whatever a kept listing still shows.
"""

from prisma.enums import ContentType
from prisma.models import (
    StoreListing,
    StoreListingReview,
    StoreListingVersion,
    UnifiedContentEmbedding,
)
from prisma.types import StoreListingWhereInput

from backend.data.account_deletion.models import StepReport, StepStatus, Subject


async def delete_marketplace(subject: Subject, *, dry_run: bool) -> StepReport:
    uid = subject.user_id
    # The listings handed over are excluded by ID as well as by owner, so a
    # dry run (nothing handed over yet) counts the same as the apply.
    listings_where: StoreListingWhereInput = {
        "owningUserId": uid,
        "id": {"not_in": subject.kept_listing_ids},
    }
    versions = await StoreListingVersion.prisma().find_many(
        where={"StoreListing": {"is": listings_where}}
    )
    version_ids = [version.id for version in versions]
    embeddings_where = {
        "contentType": ContentType.STORE_AGENT,
        "contentId": {"in": version_ids},
    }
    counts = {
        "reviews": await StoreListingReview.prisma().count(
            where={"reviewByUserId": uid}
        ),
        "listings": await StoreListing.prisma().count(where=listings_where),
        "listing_versions": len(version_ids),
        "listing_embeddings": await UnifiedContentEmbedding.prisma().count(
            where=embeddings_where
        ),
    }
    if not dry_run:
        await StoreListingReview.prisma().delete_many(where={"reviewByUserId": uid})
        await UnifiedContentEmbedding.prisma().delete_many(where=embeddings_where)
        await StoreListing.prisma().delete_many(where=listings_where)
    return StepReport(
        name="marketplace",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts=counts,
    )
