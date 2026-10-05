"""Safely rewrite legacy private-bucket media URLs to authenticated API paths.

The default mode is a read-only dry run. ``--apply`` executes owner-proven
rewrites in one transaction and protects every row with compare-and-swap.
Only exact GCS URLs below ``users/<user>/(images|videos)/<filename>`` in the
configured private bucket are eligible. Output is aggregate-only and never
contains URLs, object names, row IDs, or user IDs.

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
    from scripts.media_url_backfill import Candidate, print_report, process_candidates
else:
    from media_url_backfill import Candidate, print_report, process_candidates

APPLY_TRANSACTION_TIMEOUT = timedelta(minutes=5)

CANDIDATE_QUERY = """
WITH listing_versions AS (
    SELECT
        slv.*,
        sl."owningUserId" AS owner_user_id,
        (
            sl."activeVersionId" = slv.id
            AND slv."submissionStatus" = 'APPROVED'
            AND NOT sl."isDeleted"
            AND sl."hasApprovedVersion"
            AND NOT slv."isDeleted"
            AND slv."isAvailable"
        ) AS active_public
    FROM platform."StoreListingVersion" slv
    JOIN platform."StoreListing" sl ON sl.id = slv."storeListingId"
), candidates AS (
    SELECT
        'Profile.avatarUrl' AS target,
        p.id AS record_id,
        p."userId" AS owner_user_id,
        ARRAY[p."avatarUrl"]::text[] AS values,
        false AS is_array,
        CASE WHEN EXISTS (
            SELECT 1
            FROM platform."StoreListing" sl
            JOIN platform."StoreListingVersion" active
              ON active.id = sl."activeVersionId"
            WHERE sl."owningUserId" = p."userId"
              AND active."submissionStatus" = 'APPROVED'
              AND NOT sl."isDeleted"
              AND sl."hasApprovedVersion"
              AND NOT active."isDeleted"
              AND active."isAvailable"
        ) THEN 'active_public' END AS hold_reason
    FROM platform."Profile" p
    WHERE p."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'Expert.avatarUrl', e.id, e."ownerUserId",
        ARRAY[e."avatarUrl"]::text[], false,
        CASE WHEN e."ownerUserId" IS NULL THEN 'ambiguous' END
    FROM platform."Expert" e
    WHERE e."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'LibraryAgent.imageUrl', la.id, la."userId",
        ARRAY[la."imageUrl"]::text[], false,
        CASE WHEN NOT la."isCreatedByUser" OR la."creatorId" IS NOT NULL
             THEN 'marketplace' END
    FROM platform."LibraryAgent" la
    WHERE la."imageUrl" IS NOT NULL

    UNION ALL
    SELECT
        'StoreListingVersion.imageUrls', lv.id, lv.owner_user_id,
        lv."imageUrls", true,
        CASE WHEN lv.active_public THEN 'active_public' END
    FROM listing_versions lv
    WHERE cardinality(lv."imageUrls") > 0

    UNION ALL
    SELECT
        'StoreListingVersion.videoUrl', lv.id, lv.owner_user_id,
        ARRAY[lv."videoUrl"]::text[], false,
        CASE WHEN lv.active_public THEN 'active_public' END
    FROM listing_versions lv
    WHERE lv."videoUrl" IS NOT NULL

    UNION ALL
    SELECT
        'StoreListingVersion.agentOutputDemoUrl', lv.id, lv.owner_user_id,
        ARRAY[lv."agentOutputDemoUrl"]::text[], false,
        CASE WHEN lv.active_public THEN 'active_public' END
    FROM listing_versions lv
    WHERE lv."agentOutputDemoUrl" IS NOT NULL

    UNION ALL
    SELECT
        'Organization.avatarUrl', o.id, NULL,
        ARRAY[o."avatarUrl"]::text[], false, 'ambiguous'
    FROM platform."Organization" o
    WHERE o."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'OrganizationProfile.avatarUrl', op."organizationId", NULL,
        ARRAY[op."avatarUrl"]::text[], false, 'ambiguous'
    FROM platform."OrganizationProfile" op
    WHERE op."avatarUrl" IS NOT NULL

    UNION ALL
    SELECT
        'OAuthApplication.logoUrl', oa.id, NULL,
        ARRAY[oa."logoUrl"]::text[], false, 'ambiguous'
    FROM platform."OAuthApplication" oa
    WHERE oa."logoUrl" IS NOT NULL

    UNION ALL
    SELECT
        'SkillListingVersion.sourceUrl', sv.id, NULL,
        ARRAY[sv."sourceUrl"]::text[], false, 'ambiguous'
    FROM platform."SkillListingVersion" sv
    WHERE sv."sourceUrl" IS NOT NULL
)
SELECT target, record_id, owner_user_id, values, is_array, hold_reason
FROM candidates
WHERE EXISTS (
    SELECT 1 FROM unnest(values) AS candidate_url
    WHERE position($1 IN candidate_url) > 0
)
"""


async def main(*, apply: bool) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    backend_root = str(Path(__file__).resolve().parent.parent)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    from backend.data.db import connect, disconnect, transaction
    from backend.util.settings import Settings

    bucket = Settings().config.resolved_private_user_data_bucket
    if not bucket:
        raise SystemExit(
            "PRIVATE_USER_DATA_BUCKET or legacy MEDIA_GCS_BUCKET_NAME must be set"
        )

    await connect()
    try:
        if apply:
            async with transaction(timeout=APPLY_TRANSACTION_TIMEOUT) as client:
                report = await _run(client, bucket, apply=True)
        else:
            report = await _run(get_client(), bucket, apply=False)
    except Exception:
        raise SystemExit(
            "Backfill failed without logging identifiers; any active transaction "
            "was rolled back"
        ) from None
    finally:
        await disconnect()

    print_report(report, apply=apply)
    return 2 if report.cas_conflicts else 0


async def _run(client: Prisma, bucket: str, *, apply: bool):
    rows = await client.query_raw(CANDIDATE_QUERY, bucket)
    candidates = [Candidate.model_validate(row) for row in rows]
    return await process_candidates(client, candidates, bucket, apply=apply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply owner-proven rewrites. Without this flag the script is read-only.",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(apply=args.apply)))
