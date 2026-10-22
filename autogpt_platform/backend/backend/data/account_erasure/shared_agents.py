"""Agents other accounts still use, handed to the "deleted user" account
(`deleted_user.py`) instead of deleted, so the deletion breaks nothing of
theirs.

An agent is shared when another account has it in its library or presets, has
run it, lists it, or an expert of theirs (or a roster template) installed its
listing. So is an agent in an org another member is active in, at the org's
home or in one of its teams: every member of the org (or of the team) can
open it (`tenancy.visibility_filter`) without a row of their own pointing at
it. It keeps its org and team. Its own listing alone does not make it shared:
closing the account took every listing of its own down, so nobody can add it
any more.

Deleting a shared agent would take their library entries, presets and runs
with it, and the database refuses while one of their rows still points at
it. Every version of a shared agent goes over, since their library entries
and runs pin versions; and so does every agent one of those runs as a
sub-agent, at any depth, because `get_sub_graphs` only loads sub-agents
owned by the owner of the agent it was asked for, the top one. The listings
of shared agents go over with them, renamed to a free slug if the placeholder
already has one, and stay taken down: closing the account took every listing
of its own down (`close_account`), and nothing here puts one back up. Agents
nobody else uses are deleted with the account.
"""

import logging
from datetime import timedelta

from prisma import Prisma
from prisma.models import Profile, StoreListing
from pydantic import BaseModel

from backend.api.features.store.cache import clear_all_caches as clear_store_caches
from backend.blocks.agent import AgentExecutorBlock
from backend.data.account_erasure.deleted_user import (
    DELETED_USER_ID,
    ensure_deleted_user,
)
from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.data.db import execute_raw_with_schema, query_raw_with_schema, transaction

logger = logging.getLogger(__name__)

# $1 = the account, $2 = the sub-agent block's ID.
_SHARED_GRAPHS = """
    WITH RECURSIVE shared("id") AS (
        SELECT g."id"
          FROM {schema_prefix}"AgentGraph" g
         WHERE g."userId" = $1
           AND (
             EXISTS (SELECT 1 FROM {schema_prefix}"LibraryAgent" la
                      WHERE la."agentGraphId" = g."id" AND la."userId" <> $1)
             OR EXISTS (SELECT 1 FROM {schema_prefix}"AgentPreset" p
                         WHERE p."agentGraphId" = g."id" AND p."userId" <> $1)
             OR EXISTS (SELECT 1 FROM {schema_prefix}"AgentGraphExecution" e
                         WHERE e."agentGraphId" = g."id" AND e."userId" <> $1)
             OR EXISTS (SELECT 1 FROM {schema_prefix}"StoreListingVersion" v
                          JOIN {schema_prefix}"StoreListing" l
                            ON l."id" = v."storeListingId"
                         WHERE v."agentGraphId" = g."id"
                           AND l."owningUserId" <> $1)
             OR EXISTS (SELECT 1 FROM {schema_prefix}"StoreListingVersion" v
                          JOIN {schema_prefix}"ExpertWorkflow" w
                            ON w."storeListingVersionId" = v."id"
                          JOIN {schema_prefix}"Expert" x ON x."id" = w."expertId"
                         WHERE v."agentGraphId" = g."id"
                           AND x."ownerUserId" IS DISTINCT FROM $1)
             -- Org-home and team agents: seen by the org's (or team's)
             -- members, so shared while anyone else is active in the org.
             -- Orgs nobody else is active in go with the account
             -- (`preflight._orgs`), and their agents with them.
             OR EXISTS (SELECT 1 FROM {schema_prefix}"OrgMember" m
                         WHERE m."orgId" = g."organizationId"
                           AND m."userId" <> $1 AND m."status" = 'ACTIVE')
             OR EXISTS (SELECT 1 FROM {schema_prefix}"Team" t
                          JOIN {schema_prefix}"OrgMember" m ON m."orgId" = t."orgId"
                         WHERE t."id" = g."teamId"
                           AND m."userId" <> $1 AND m."status" = 'ACTIVE')
           )
        UNION
        SELECT sub."id"
          FROM shared s
          JOIN {schema_prefix}"AgentNode" n
            ON n."agentGraphId" = s."id" AND n."agentBlockId" = $2
          JOIN {schema_prefix}"AgentGraph" sub
            ON sub."id" = n."constantInput"->>'graph_id' AND sub."userId" = $1
    )
    SELECT DISTINCT "id" FROM shared ORDER BY "id"
"""
_KEPT_LISTINGS = """
    SELECT l."id" FROM {schema_prefix}"StoreListing" l
     WHERE l."owningUserId" = $1 AND l."agentGraphId" = ANY($2::text[])
     ORDER BY l."id"
"""
_TRANSFER_GRAPHS = """
    UPDATE {schema_prefix}"AgentGraph" SET "userId" = $1
     WHERE "userId" = $2 AND "id" = ANY($3::text[])
"""
_TRANSFER_LISTING = """
    UPDATE {schema_prefix}"StoreListing" SET "owningUserId" = $1, "slug" = $2
     WHERE "id" = $3 AND "owningUserId" = $4
"""
# A submission still waiting for review would otherwise be published under
# the placeholder after its author asked to be deleted. REJECTED still counts
# as submitted, so nobody who has that version in their library loses it.
_WITHDRAW_PENDING = """
    UPDATE {schema_prefix}"StoreListingVersion" SET "submissionStatus" = 'REJECTED'
     WHERE "storeListingId" = ANY($1::text[]) AND "submissionStatus" = 'PENDING'
"""
_RELABEL_LIBRARY = """
    UPDATE {schema_prefix}"LibraryAgent" SET "creatorId" = $1
     WHERE "creatorId" = $2 AND "userId" <> $3
"""
_COUNT_RELABEL = """
    SELECT COUNT(*)::int AS n FROM {schema_prefix}"LibraryAgent" la
      JOIN {schema_prefix}"Profile" p ON p."id" = la."creatorId"
     WHERE p."userId" = $1 AND la."userId" <> $1
"""
_TRANSACTION_TIMEOUT = timedelta(minutes=2)


class TransferPlan(BaseModel):
    graph_ids: list[str] = []
    listing_ids: list[str] = []


async def plan_transfer(user_id: str) -> TransferPlan:
    """Read-only: which of the account's agents and listings go over."""
    rows = await query_raw_with_schema(_SHARED_GRAPHS, user_id, AgentExecutorBlock().id)
    graph_ids = [row["id"] for row in rows]
    if not graph_ids:
        return TransferPlan()
    listings = await query_raw_with_schema(_KEPT_LISTINGS, user_id, graph_ids)
    return TransferPlan(
        graph_ids=graph_ids, listing_ids=[row["id"] for row in listings]
    )


async def transfer_shared_agents(subject: Subject, *, dry_run: bool) -> StepReport:
    if not subject.shared_graph_ids:
        return _report({"agents": 0, "listings": 0}, dry_run)
    counts = {
        "agents": len(subject.shared_graph_ids),
        "listings": len(subject.kept_listing_ids),
        "library_entries_relabelled": await _count_relabel(subject.user_id),
    }
    if not dry_run:
        await _transfer(subject)
        _clear_store_caches()
    return _report(counts, dry_run)


def _clear_store_caches() -> None:
    """So the marketplace shows the new creator now rather than within the
    caches' five minutes. Best effort: the transfer is already committed, and
    a rerun has nothing left to transfer, so failing the step would help
    nobody."""
    try:
        clear_store_caches()
    except Exception:
        logger.warning("Could not clear the store caches", exc_info=True)


async def _transfer(subject: Subject) -> None:
    uid = subject.user_id
    async with transaction(timeout=_TRANSACTION_TIMEOUT) as tx:
        placeholder_profile = await ensure_deleted_user(tx)
        await execute_raw_with_schema(
            _TRANSFER_GRAPHS, DELETED_USER_ID, uid, subject.shared_graph_ids, client=tx
        )
        listings = await StoreListing.prisma(tx).find_many(
            where={"id": {"in": subject.kept_listing_ids}, "owningUserId": uid}
        )
        for listing in listings:
            slug = await _free_slug(tx, listing.slug, listing.id)
            await execute_raw_with_schema(
                _TRANSFER_LISTING, DELETED_USER_ID, slug, listing.id, uid, client=tx
            )
        if subject.kept_listing_ids:
            await execute_raw_with_schema(
                _WITHDRAW_PENDING, subject.kept_listing_ids, client=tx
            )
        own_profile = await Profile.prisma(tx).find_unique(where={"userId": uid})
        if own_profile:
            await execute_raw_with_schema(
                _RELABEL_LIBRARY, placeholder_profile, own_profile.id, uid, client=tx
            )


async def _free_slug(tx: Prisma, slug: str, listing_id: str) -> str:
    """The listing's slug, or the first `slug-N` the placeholder has free:
    its listings share one creator, and slugs are unique per creator."""
    candidate, n = slug, 2
    while await StoreListing.prisma(tx).find_first(
        where={
            "owningUserId": DELETED_USER_ID,
            "slug": candidate,
            "NOT": [{"id": listing_id}],
        }
    ):
        candidate, n = f"{slug}-{n}", n + 1
    return candidate


async def _count_relabel(user_id: str) -> int:
    rows = await query_raw_with_schema(_COUNT_RELABEL, user_id)
    return int(rows[0]["n"]) if rows else 0


def _report(counts: dict[str, int], dry_run: bool) -> StepReport:
    return StepReport(
        name="shared_agents",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts=counts,
    )
