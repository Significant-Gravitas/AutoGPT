"""The account's marketplace presence: its reviews and its listings.

Both reference the account with ON DELETE RESTRICT, so they must go before
the account's rows can. Listing versions take their submissions and the
reviews other accounts left on them along; their search embeddings carry no
foreign key and are deleted here. The listings' media, public copies
included, lives under the account's own storage prefix (`storage.py`).
"""

from prisma.enums import ContentType
from prisma.models import (
    StoreListing,
    StoreListingReview,
    StoreListingVersion,
    UnifiedContentEmbedding,
)

from backend.data.account_deletion.models import StepReport, StepStatus, Subject
from backend.data.db import execute_raw_with_schema

# Other accounts' library entries and presets on this account's agents. Only
# run when the operator passed --remove-shared-agents (`preflight.py`).
_REMOVE_SHARED_LIBRARY_ENTRIES = """
    DELETE FROM {schema_prefix}"LibraryAgent" la
     USING {schema_prefix}"AgentGraph" g
     WHERE g."id" = la."agentGraphId" AND g."version" = la."agentGraphVersion"
       AND g."userId" = $1 AND la."userId" <> $1
"""
_REMOVE_SHARED_PRESETS = """
    DELETE FROM {schema_prefix}"AgentPreset" p
     USING {schema_prefix}"AgentGraph" g
     WHERE g."id" = p."agentGraphId" AND g."version" = p."agentGraphVersion"
       AND g."userId" = $1 AND p."userId" <> $1
"""


async def delete_marketplace(
    subject: Subject,
    *,
    dry_run: bool,
    shared_agent_rows: int = 0,
    remove_shared_agents: bool = False,
) -> StepReport:
    uid = subject.user_id
    versions = await StoreListingVersion.prisma().find_many(
        where={"StoreListing": {"is": {"owningUserId": uid}}}
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
        "listings": await StoreListing.prisma().count(where={"owningUserId": uid}),
        "listing_versions": len(version_ids),
        "listing_embeddings": await UnifiedContentEmbedding.prisma().count(
            where=embeddings_where
        ),
        "shared_agent_rows": shared_agent_rows if remove_shared_agents else 0,
    }
    if not dry_run:
        await StoreListingReview.prisma().delete_many(where={"reviewByUserId": uid})
        await UnifiedContentEmbedding.prisma().delete_many(where=embeddings_where)
        await StoreListing.prisma().delete_many(where={"owningUserId": uid})
        if remove_shared_agents:
            await execute_raw_with_schema(_REMOVE_SHARED_LIBRARY_ENTRIES, uid)
            await execute_raw_with_schema(_REMOVE_SHARED_PRESETS, uid)
    return StepReport(
        name="marketplace",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts=counts,
    )
