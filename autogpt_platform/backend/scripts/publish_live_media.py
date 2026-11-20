"""Publish media that the public site already serves from the legacy bucket.

Before public access on the legacy bucket is revoked, every reference the
public site serves must point at the public site media bucket. This copies
each such object from the private (legacy) bucket to the public bucket under
the same object path and rewrites the reference with compare-and-swap:

- media of every approved, non-deleted version on a non-deleted listing;
- avatars of creators with such a listing or a live skill listing;
- library copies of a published listing image;
- OAuth application logos stored under ``oauth-apps/<app id>/logo/``.

Only ``users/<owner>/(images|videos)/<file>`` objects owned by the row's owner
(or, for listing media, an active member of the listing's organization) and
logos under the app's own prefix are copied. The default mode is a read-only
dry run; ``--apply`` copies and writes in short transactions. Already-public
references are skipped, so the script is safe to re-run and repairs a failed
approval-time copy. Output is aggregate-only and never contains URLs, object
names, row IDs, or user IDs.

Usage::

    poetry run python scripts/publish_live_media.py
    poetry run python scripts/publish_live_media.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from pathlib import Path

from prisma import Prisma, get_client

if __package__:
    from scripts.media_url_backfill import (
        PRIVATE_MEDIA_PREFIX,
        ApplyProgress,
        Target,
        Transaction,
        apply_mutations,
        resolve_source_bucket,
    )
    from scripts.media_url_backfill_queries import (
        CREATOR_IS_PUBLIC,
        PUBLISH_UPDATE_QUERIES,
        VERSION_IS_PUBLIC,
    )
    from scripts.media_url_publish import (
        Buckets,
        ObjectCopier,
        PublishCandidate,
        PublishOutcome,
        PublishReport,
        build_mutations,
        copy_objects,
        library_mutations,
        plan_publication,
        print_report,
        published_image_paths,
    )
else:
    from media_url_backfill import (
        PRIVATE_MEDIA_PREFIX,
        ApplyProgress,
        Target,
        Transaction,
        apply_mutations,
        resolve_source_bucket,
    )
    from media_url_backfill_queries import (
        CREATOR_IS_PUBLIC,
        PUBLISH_UPDATE_QUERIES,
        VERSION_IS_PUBLIC,
    )
    from media_url_publish import (
        Buckets,
        ObjectCopier,
        PublishCandidate,
        PublishOutcome,
        PublishReport,
        build_mutations,
        copy_objects,
        library_mutations,
        plan_publication,
        print_report,
        published_image_paths,
    )

BATCH_TRANSACTION_TIMEOUT = timedelta(seconds=30)

BLOCKING_OUTCOMES = (
    PublishOutcome.COPY_FAILED,
    PublishOutcome.SKIP_FOREIGN_OWNER,
    PublishOutcome.SKIP_MALFORMED,
    PublishOutcome.SKIP_UNRECOGNIZED,
)

CopierFactory = Callable[[], AbstractAsyncContextManager[ObjectCopier]]

LISTING_QUERY = f"""
SELECT
    slv.id AS record_id,
    slv."imageUrls" AS image_urls,
    slv."videoUrl" AS video_url,
    slv."agentOutputDemoUrl" AS demo_url,
    ARRAY[sl."owningUserId"] || ARRAY(
        SELECT om."userId"
        FROM platform."OrgMember" AS om
        WHERE om."orgId" = sl."owningOrgId"
          AND om.status = 'ACTIVE'
    ) AS owner_ids
FROM platform."StoreListingVersion" AS slv
JOIN platform."StoreListing" AS sl ON sl.id = slv."storeListingId"
WHERE {VERSION_IS_PUBLIC}
"""

PROFILE_QUERY = f"""
SELECT p.id AS record_id, p."userId" AS owner_id, p."avatarUrl" AS value
FROM platform."Profile" AS p
WHERE p."avatarUrl" IS NOT NULL
  AND {CREATOR_IS_PUBLIC}
"""

OAUTH_LOGO_QUERY = """
SELECT oa.id AS record_id, oa."logoUrl" AS value
FROM platform."OAuthApplication" AS oa
WHERE oa."logoUrl" IS NOT NULL
"""

LIBRARY_QUERY = """
SELECT la.id AS record_id, la."imageUrl" AS value
FROM platform."LibraryAgent" AS la
WHERE la."imageUrl" IS NOT NULL
  AND EXISTS (
      SELECT 1 FROM unnest($1::text[]) AS needle
      WHERE position(needle IN la."imageUrl") > 0
  )
"""


async def main(*, apply: bool, bucket_override: str | None = None) -> int:
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL must be set")

    backend_root = str(Path(__file__).resolve().parent.parent)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    from gcloud.aio import storage as async_storage

    from backend.data.db import connect, disconnect, transaction
    from backend.util.settings import Settings

    config = Settings().config
    private_bucket = config.resolved_private_user_data_bucket
    buckets = Buckets(
        source=resolve_source_bucket(
            private_bucket=private_bucket,
            legacy_bucket=config.media_gcs_bucket_name,
            override=bucket_override,
        ),
        private=private_bucket,
        public=config.public_site_media_bucket,
    )
    if not buckets.public:
        raise SystemExit("PUBLIC_SITE_MEDIA_BUCKET must be set")
    if buckets.public in {buckets.source, buckets.private}:
        raise SystemExit(
            "PUBLIC_SITE_MEDIA_BUCKET must differ from the private and legacy buckets"
        )

    def batch_transaction():
        return transaction(timeout=BATCH_TRANSACTION_TIMEOUT)

    progress = ApplyProgress()
    await connect()
    try:
        report = await _run(
            get_client(),
            buckets,
            apply=apply,
            transaction=batch_transaction,
            copier_factory=async_storage.Storage,
            progress=progress,
        )
    except Exception:
        raise SystemExit(
            "Publication failed without logging identifiers after "
            f"{progress.applied_rows} committed row updates; the batch in "
            "progress was rolled back. Re-running is safe."
        ) from None
    finally:
        await disconnect()

    print_report(report, apply=apply)
    return 2 if report.cas_conflicts or unpublished_references(report) else 0


def unpublished_references(report: PublishReport) -> int:
    """Live references that would break once the legacy bucket goes private."""
    return sum(
        counts[outcome]
        for counts in report.counts.values()
        for outcome in BLOCKING_OUTCOMES
    )


async def _run(
    client: Prisma,
    buckets: Buckets,
    *,
    apply: bool,
    transaction: Transaction,
    copier_factory: CopierFactory,
    progress: ApplyProgress,
) -> PublishReport:
    plan = plan_publication(await _load_candidates(client), buckets)
    objects = plan.objects
    if apply and objects:
        async with copier_factory() as copier:
            copied = await copy_objects(objects, buckets.public, copier)
    else:
        copied = objects
    mutations, counts = build_mutations(plan, copied, buckets.public)

    library_rows = await client.query_raw(
        LIBRARY_QUERY, [buckets.source, buckets.private, PRIVATE_MEDIA_PREFIX]
    )
    mutations += library_mutations(
        ((row["record_id"], row["value"]) for row in library_rows),
        published_image_paths(plan, copied),
        buckets,
        counts,
    )

    report = PublishReport(
        counts=counts,
        planned_objects=len(objects),
        copied_objects=len(copied) if apply else 0,
        planned_rows=len(mutations),
    )
    if apply:
        await apply_mutations(mutations, PUBLISH_UPDATE_QUERIES, transaction, progress)
        report.applied_rows = progress.applied_rows
        report.cas_conflicts = progress.cas_conflicts
    return report


async def _load_candidates(client: Prisma) -> list[PublishCandidate]:
    candidates: list[PublishCandidate] = []
    for row in await client.query_raw(LISTING_QUERY):
        common = {"record_id": row["record_id"], "owner_ids": row["owner_ids"]}
        if row["image_urls"]:
            candidates.append(
                PublishCandidate(
                    target=Target.LISTING_IMAGES,
                    values=row["image_urls"],
                    is_array=True,
                    **common,
                )
            )
        for target, column in (
            (Target.LISTING_VIDEO, "video_url"),
            (Target.LISTING_DEMO, "demo_url"),
        ):
            if row[column]:
                candidates.append(
                    PublishCandidate(
                        target=target, values=[row[column]], is_array=False, **common
                    )
                )
    for row in await client.query_raw(PROFILE_QUERY):
        candidates.append(
            PublishCandidate(
                target=Target.PROFILE_AVATAR,
                record_id=row["record_id"],
                values=[row["value"]],
                is_array=False,
                owner_ids=[row["owner_id"]],
            )
        )
    for row in await client.query_raw(OAUTH_LOGO_QUERY):
        candidates.append(
            PublishCandidate(
                target=Target.OAUTH_LOGO,
                record_id=row["record_id"],
                values=[row["value"]],
                is_array=False,
            )
        )
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy objects and rewrite references. Without this flag the script "
        "is read-only.",
    )
    parser.add_argument(
        "--bucket",
        help="Legacy bucket the stored URLs point at. Defaults to "
        "PRIVATE_USER_DATA_BUCKET (or MEDIA_GCS_BUCKET_NAME).",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(apply=args.apply, bucket_override=args.bucket)))
