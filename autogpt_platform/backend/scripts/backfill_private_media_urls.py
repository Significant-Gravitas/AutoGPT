"""Safely rewrite legacy private-bucket media URLs to authenticated API paths.

The default mode is a read-only dry run. ``--apply`` executes the rewrites in
short transactions of a few hundred rows and protects every row with
compare-and-swap, so an interrupted run keeps its progress and a re-run picks
up where it stopped. Only GCS URLs below ``users/<user>/(images|videos)/<file>``
in the legacy bucket are eligible; references that are served publicly are
held, so run ``publish_live_media.py --apply`` first. Output is aggregate-only
and never contains URLs, object names, row IDs, or user IDs.

Usage::

    poetry run python scripts/backfill_private_media_urls.py
    poetry run python scripts/backfill_private_media_urls.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import timedelta
from pathlib import Path

from prisma import Prisma, get_client

if __package__:
    from scripts.media_url_backfill import (
        ApplyProgress,
        BackfillReport,
        Candidate,
        Outcome,
        Transaction,
        print_report,
        process_candidates,
        resolve_source_bucket,
    )
    from scripts.media_url_backfill_queries import CREATOR_IS_PUBLIC, VERSION_IS_PUBLIC
else:
    from media_url_backfill import (
        ApplyProgress,
        BackfillReport,
        Candidate,
        Outcome,
        Transaction,
        print_report,
        process_candidates,
        resolve_source_bucket,
    )
    from media_url_backfill_queries import CREATOR_IS_PUBLIC, VERSION_IS_PUBLIC

BATCH_TRANSACTION_TIMEOUT = timedelta(seconds=30)

CANDIDATE_QUERY = f"""
WITH candidates AS (
    SELECT
        'Profile.avatarUrl' AS target,
        p.id AS record_id,
        p."userId" AS owner_user_id,
        ARRAY[p."avatarUrl"]::text[] AS values,
        false AS is_array,
        CASE WHEN {CREATOR_IS_PUBLIC} THEN 'public' END AS hold_reason,
        ARRAY[]::text[] AS co_owner_ids
    FROM platform."Profile" AS p
    WHERE p."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'Expert.avatarUrl', e.id, e."ownerUserId",
        ARRAY[e."avatarUrl"]::text[], false,
        CASE WHEN e."ownerUserId" IS NULL THEN 'ambiguous' END, ARRAY[]::text[]
    FROM platform."Expert" AS e
    WHERE e."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'LibraryAgent.imageUrl', la.id, la."userId",
        ARRAY[la."imageUrl"]::text[], false, NULL,
        ARRAY(
            SELECT colleague."userId"
            FROM platform."OrgMember" AS me
            JOIN platform."OrgMember" AS colleague ON colleague."orgId" = me."orgId"
            JOIN platform."Organization" AS org ON org.id = me."orgId"
            WHERE me."userId" = la."userId"
              AND me.status = 'ACTIVE'
              AND colleague.status = 'ACTIVE'
              AND org."deletedAt" IS NULL
        )
    FROM platform."LibraryAgent" AS la
    WHERE la."imageUrl" IS NOT NULL

    UNION ALL
    SELECT
        'StoreListingVersion.imageUrls', slv.id, sl."owningUserId",
        slv."imageUrls", true,
        CASE WHEN {VERSION_IS_PUBLIC} THEN 'public' END,
        ARRAY(
            SELECT om."userId"
            FROM platform."OrgMember" AS om
            WHERE om."orgId" = sl."owningOrgId" AND om.status = 'ACTIVE'
        )
    FROM platform."StoreListingVersion" AS slv
    JOIN platform."StoreListing" AS sl ON sl.id = slv."storeListingId"
    WHERE cardinality(slv."imageUrls") > 0

    UNION ALL
    SELECT
        'StoreListingVersion.videoUrl', slv.id, sl."owningUserId",
        ARRAY[slv."videoUrl"]::text[], false,
        CASE WHEN {VERSION_IS_PUBLIC} THEN 'public' END,
        ARRAY(
            SELECT om."userId"
            FROM platform."OrgMember" AS om
            WHERE om."orgId" = sl."owningOrgId" AND om.status = 'ACTIVE'
        )
    FROM platform."StoreListingVersion" AS slv
    JOIN platform."StoreListing" AS sl ON sl.id = slv."storeListingId"
    WHERE slv."videoUrl" IS NOT NULL

    UNION ALL
    SELECT
        'StoreListingVersion.agentOutputDemoUrl', slv.id, sl."owningUserId",
        ARRAY[slv."agentOutputDemoUrl"]::text[], false,
        CASE WHEN {VERSION_IS_PUBLIC} THEN 'public' END,
        ARRAY(
            SELECT om."userId"
            FROM platform."OrgMember" AS om
            WHERE om."orgId" = sl."owningOrgId" AND om.status = 'ACTIVE'
        )
    FROM platform."StoreListingVersion" AS slv
    JOIN platform."StoreListing" AS sl ON sl.id = slv."storeListingId"
    WHERE slv."agentOutputDemoUrl" IS NOT NULL

    UNION ALL
    SELECT
        'Organization.avatarUrl', o.id, NULL,
        ARRAY[o."avatarUrl"]::text[], false, NULL,
        ARRAY(
            SELECT om."userId"
            FROM platform."OrgMember" AS om
            WHERE om."orgId" = o.id AND om.status = 'ACTIVE'
        )
    FROM platform."Organization" AS o
    WHERE o."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'OrganizationProfile.avatarUrl', op."organizationId", NULL,
        ARRAY[op."avatarUrl"]::text[], false, NULL,
        ARRAY(
            SELECT om."userId"
            FROM platform."OrgMember" AS om
            WHERE om."orgId" = op."organizationId" AND om.status = 'ACTIVE'
        )
    FROM platform."OrganizationProfile" AS op
    WHERE op."avatarUrl" IS NOT NULL
)
SELECT target, record_id, owner_user_id, values, is_array, hold_reason, co_owner_ids
FROM candidates
WHERE EXISTS (
    SELECT 1 FROM unnest(values) AS candidate_url
    WHERE position($1 IN candidate_url) > 0
)
"""


async def main(*, apply: bool, bucket_override: str | None = None) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    backend_root = str(Path(__file__).resolve().parent.parent)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    from backend.data.db import connect, disconnect, transaction
    from backend.util.settings import Settings

    config = Settings().config
    private_bucket = config.resolved_private_user_data_bucket
    bucket = resolve_source_bucket(
        private_bucket=private_bucket,
        legacy_bucket=config.media_gcs_bucket_name,
        override=bucket_override,
    )
    if bucket != private_bucket:
        print(
            "Warning: --bucket differs from PRIVATE_USER_DATA_BUCKET. Rewritten "
            "references are served from PRIVATE_USER_DATA_BUCKET, so their "
            "objects must already exist there under the same paths."
        )
    public_bucket = config.public_site_media_bucket
    if public_bucket == bucket:
        public_bucket = ""

    def batch_transaction():
        return transaction(timeout=BATCH_TRANSACTION_TIMEOUT)

    progress = ApplyProgress()
    await connect()
    try:
        report = await _run(
            get_client(),
            bucket,
            public_bucket,
            apply=apply,
            transaction=batch_transaction,
            progress=progress,
        )
    except Exception:
        raise SystemExit(
            "Backfill failed without logging identifiers after "
            f"{progress.applied_rows} committed row updates; the batch in "
            "progress was rolled back. Re-running is safe."
        ) from None
    finally:
        await disconnect()

    print_report(report, apply=apply)
    return 2 if report.cas_conflicts or stranded_references(report) else 0


def stranded_references(report: BackfillReport) -> int:
    """References left on the legacy bucket, which stop loading once it is private."""
    return sum(
        count
        for outcome, count in report.counts.items()
        if outcome not in {Outcome.REWRITE, Outcome.ALREADY_PUBLIC}
    )


async def _run(
    client: Prisma,
    bucket: str,
    public_bucket: str,
    *,
    apply: bool,
    transaction: Transaction,
    progress: ApplyProgress,
) -> BackfillReport:
    rows = await client.query_raw(CANDIDATE_QUERY, bucket)
    candidates = [Candidate.model_validate(row) for row in rows]
    return await process_candidates(
        candidates,
        bucket,
        public_bucket=public_bucket or None,
        apply=apply,
        transaction=transaction,
        progress=progress,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply owner-proven rewrites. Without this flag the script is read-only.",
    )
    parser.add_argument(
        "--bucket",
        help="Legacy bucket the stored URLs point at. Defaults to "
        "PRIVATE_USER_DATA_BUCKET (or MEDIA_GCS_BUCKET_NAME).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(apply=args.apply, bucket_override=args.bucket)))
