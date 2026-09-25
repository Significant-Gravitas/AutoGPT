"""Publish the skills catalog into the marketplace.

The catalog (see :mod:`skill_catalog_release`) is the source of every
platform skill and of the expert roster. This module turns one loaded commit
into marketplace rows. The rules that make updates safe:

* A version is content-addressed. Its ``packageSha256`` is the catalog's
  ``tree_sha256`` over the sorted ``(path, sha256, executable)`` list,
  SKILL.md included, and once a version carries a hash nothing rewrites its
  content. Publishing a package whose hash the listing already has re-uses
  that row; anything else inserts a new version and moves ``activeVersionId``.
* Rollback is publishing an older commit. The hashes are already there.
* Only platform rows are written: listings with both owner columns null, and
  templates with no owner. A user's copy is never touched here. What reaches
  copies is :mod:`backend.copilot.tools.skills`, lazily, by comparing a copy's
  recorded baseline to the active version.
* A package that disappears from the catalog without an entry in
  ``retirements`` is left alone and reported. Absence is not deletion.

Run by ``publish-skills-catalog`` right after ``prisma migrate deploy`` in
every environment, so a deploy publishes the catalog's ``main``.
"""

import asyncio
import logging
from collections.abc import Iterable
from datetime import timedelta

import prisma
import prisma.enums
import prisma.models
import prisma.types
from pydantic import BaseModel

from backend.api.features.experts import seed as expert_seed
from backend.data import db as database
from backend.util.json import SafeJson

from . import skill_db
from .skill_catalog_release import LoadedPackage, LoadedRelease
from .skill_model import legacy_package_sha256
from .skill_submission_db import snapshot_version_files

logger = logging.getLogger(__name__)

PUBLISH_TRANSACTION_TIMEOUT = timedelta(minutes=10)
_CACHE_DROP_TIMEOUT_S = 10
# Two deploys publishing at once serialize on this; the second finds nothing
# to do. hashtextextended() keys the Postgres advisory lock off the string.
_PUBLISH_LOCK_KEY = "skills-catalog:publish"

_PLATFORM_LISTING: prisma.types.SkillListingWhereInput = {
    "owningUserId": None,
    "owningOrgId": None,
}


class CatalogConflictError(ValueError):
    """A catalog slug collides with a listing a user owns."""


class PublishSummary(BaseModel):
    repository: str
    revision: str
    release_key: str
    dry_run: bool
    created: list[str] = []
    updated: list[str] = []
    unchanged: int = 0
    retired: list[str] = []
    # Live platform listings the catalog no longer mentions. Left alone.
    orphaned: list[str] = []
    # Legacy versions that got a content hash so existing copies can be matched.
    backfilled: int = 0
    experts: dict[str, list[str]] = {}
    release_id: str | None = None


async def publish_catalog(
    loaded: LoadedRelease,
    *,
    repository: str,
    revision: str,
    dry_run: bool = False,
    seed_experts: bool = True,
) -> PublishSummary:
    """Make the marketplace serve *loaded*. Idempotent: publishing the same
    release twice changes nothing the second time.

    Skills go in one transaction under an advisory lock. The roster follows
    in the expert seed's own steps, as it did before the catalog owned it.
    A dry run reads the same state and reports the plan without writing.
    """
    summary = PublishSummary(
        repository=repository,
        revision=revision,
        release_key=loaded.release_key,
        dry_run=dry_run,
    )
    async with database.transaction(timeout=PUBLISH_TRANSACTION_TIMEOUT) as tx:
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            _PUBLISH_LOCK_KEY,
        )
        summary.backfilled = await _backfill_legacy_hashes(tx, dry_run=dry_run)
        for package in loaded.packages:
            outcome = await _publish_package(tx, package, dry_run=dry_run)
            if outcome == "created":
                summary.created.append(package.slug)
            elif outcome == "updated":
                summary.updated.append(package.slug)
            else:
                summary.unchanged += 1
        summary.retired = await _retire(tx, loaded.retirements, dry_run=dry_run)
        summary.orphaned = await _orphaned(tx, loaded)
    if seed_experts:
        summary.experts = await expert_seed.seed_roster(
            loaded.experts, retired_keys=loaded.retired_experts, dry_run=dry_run
        )
    if not dry_run:
        summary.release_id = await _record_release(loaded, summary)
        await _drop_active_versions_cache()
    logger.info(
        f"{'Would publish' if dry_run else 'Published'} {loaded.release_key} "
        f"({revision[:12]}): {len(summary.created)} created, "
        f"{len(summary.updated)} updated, {summary.unchanged} unchanged, "
        f"{len(summary.retired)} retired, {len(summary.orphaned)} orphaned"
    )
    return summary


async def _drop_active_versions_cache() -> None:
    """Best-effort and bounded: the deploy job that runs the publisher has no
    Redis in reach, and the client's connect retry would otherwise hold the
    deploy for far longer than the cache's own TTL."""
    try:
        await asyncio.wait_for(
            skill_db.invalidate_active_versions_cache(), timeout=_CACHE_DROP_TIMEOUT_S
        )
    except (asyncio.TimeoutError, Exception):
        logger.warning(
            "Active skill versions cache not dropped; copies will see the "
            f"publish within {skill_db.ACTIVE_VERSIONS_CACHE_TTL_S}s"
        )


async def _record_release(loaded: LoadedRelease, summary: PublishSummary) -> str:
    release = await prisma.models.SkillCatalogRelease.prisma().create(
        data={
            "repository": summary.repository,
            "revision": summary.revision,
            "releaseKey": loaded.release_key,
            "manifestSha256": loaded.manifest_sha256,
            "summary": SafeJson(summary.model_dump()),
        }
    )
    return release.id


async def _publish_package(
    tx: prisma.Prisma, package: LoadedPackage, *, dry_run: bool
) -> str:
    listing = await prisma.models.SkillListing.prisma(tx).find_unique(
        where={"slug": package.slug}
    )
    if listing is not None and (
        listing.owningUserId is not None or listing.owningOrgId is not None
    ):
        raise CatalogConflictError(
            f"catalog skill '{package.slug}' collides with a listing a user owns"
        )
    if listing is None:
        if dry_run:
            return "created"
        listing = await prisma.models.SkillListing.prisma(tx).create(
            data={"slug": package.slug, "hasApprovedVersion": True}
        )
        outcome = "created"
    else:
        outcome = "updated"
    version = await _version_for(tx, listing.id, package, dry_run=dry_run)
    if version is None:
        return outcome
    if (
        outcome == "updated"
        and listing.activeVersionId == version.id
        and not listing.isDeleted
        and listing.hasApprovedVersion
    ):
        return "unchanged"
    if not dry_run:
        await prisma.models.SkillListing.prisma(tx).update(
            where={"id": listing.id},
            data={
                "activeVersionId": version.id,
                "isDeleted": False,
                "hasApprovedVersion": True,
            },
        )
    return outcome


async def _version_for(
    tx: prisma.Prisma, listing_id: str, package: LoadedPackage, *, dry_run: bool
) -> prisma.models.SkillListingVersion | None:
    """The listing's version with this package's hash, created if absent and
    made servable if it was withdrawn or retired. None on a dry run that
    would have created one."""
    version = await prisma.models.SkillListingVersion.prisma(tx).find_first(
        where={"skillListingId": listing_id, "packageSha256": package.package_sha256}
    )
    if version is None:
        return None if dry_run else await _create_version(tx, listing_id, package)
    withdrawn = (
        version.isDeleted
        or not version.isAvailable
        or version.submissionStatus != prisma.enums.SubmissionStatus.APPROVED
    )
    if withdrawn and not dry_run:
        await prisma.models.SkillListingVersion.prisma(tx).update(
            where={"id": version.id},
            data={
                "isDeleted": False,
                "isAvailable": True,
                "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
            },
        )
    return version


async def _create_version(
    tx: prisma.Prisma, listing_id: str, package: LoadedPackage
) -> prisma.models.SkillListingVersion:
    latest = await prisma.models.SkillListingVersion.prisma(tx).find_first(
        where={"skillListingId": listing_id}, order={"version": "desc"}
    )
    created = await prisma.models.SkillListingVersion.prisma(tx).create(
        data={
            "skillListingId": listing_id,
            "version": (latest.version + 1) if latest else 1,
            "name": package.parsed.name,
            "description": package.parsed.description,
            "body": package.parsed.body,
            "triggers": list(package.parsed.triggers),
            "categories": package.entry["categories"],
            "requiredProviders": package.entry["required_providers"],
            "sourceSkillSlug": package.slug,
            "sourceRepo": _attribution(package, "source"),
            "sourceUrl": _attribution(package, "source_url"),
            "license": _attribution(package, "license"),
            "packageSha256": package.package_sha256,
            "skillMarkdown": package.skill_markdown,
            "isAvailable": True,
            "isDeleted": False,
            "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        }
    )
    await snapshot_version_files(created.id, package.files, tx)
    return created


def _attribution(package: LoadedPackage, key: str) -> str | None:
    """Frontmatter first (``metadata.<key>`` on vendored skills, top-level on
    older ones), then the catalog entry for a vendored package. ``platform``
    is not an attribution."""
    metadata = package.parsed.extra.get("metadata")
    nested = metadata.get(key) if isinstance(metadata, dict) else None
    value = _text(nested) or _text(package.parsed.extra.get(key))
    if value is not None:
        return value
    if key == "source" and package.entry["source"] != "platform":
        return package.entry["source"]
    if key == "license":
        return package.entry["license"]
    return None


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


async def _retire(
    tx: prisma.Prisma, slugs: Iterable[str], *, dry_run: bool
) -> list[str]:
    retired: list[str] = []
    for slug in slugs:
        listing = await prisma.models.SkillListing.prisma(tx).find_unique(
            where={"slug": slug}
        )
        if (
            listing is None
            or listing.owningUserId is not None
            or listing.owningOrgId is not None
            or listing.isDeleted
        ):
            continue
        retired.append(slug)
        if dry_run:
            continue
        await prisma.models.SkillListing.prisma(tx).update(
            where={"id": listing.id}, data={"isDeleted": True}
        )
        if listing.activeVersionId is not None:
            await prisma.models.SkillListingVersion.prisma(tx).update(
                where={"id": listing.activeVersionId}, data={"isAvailable": False}
            )
    return retired


async def _orphaned(tx: prisma.Prisma, loaded: LoadedRelease) -> list[str]:
    """Live platform listings the release neither ships nor retires."""
    listings = await prisma.models.SkillListing.prisma(tx).find_many(
        where={**_PLATFORM_LISTING, "isDeleted": False}
    )
    known = loaded.slugs | set(loaded.retirements)
    return sorted(listing.slug for listing in listings if listing.slug not in known)


async def _backfill_legacy_hashes(tx: prisma.Prisma, *, dry_run: bool) -> int:
    """Give versions written before the publisher a ``packageSha256``.

    Their installs rendered ``SKILL.md`` from the row's fields, so the hash
    is over that rendering plus the files. That is what lets a copy installed
    from such a version prove it is unmodified and take the first catalog
    update automatically. A rendering that collides with a hash the listing
    already has is left null: the copy is then treated as modified, which
    only costs that user an automatic update.
    """
    versions = await prisma.models.SkillListingVersion.prisma(tx).find_many(
        where={"packageSha256": None, "SkillListing": {"is": _PLATFORM_LISTING}},
        include={"Files": True, "SkillListing": True},
    )
    if not versions:
        return 0
    hashed = await prisma.models.SkillListingVersion.prisma(tx).find_many(
        where={
            "skillListingId": {"in": sorted({v.skillListingId for v in versions})},
            "NOT": [{"packageSha256": None}],
        }
    )
    taken = {(row.skillListingId, row.packageSha256) for row in hashed}
    stamped = 0
    for version in versions:
        if version.SkillListing is None:
            continue
        digest = legacy_package_sha256(version, version.SkillListing.slug)
        if (version.skillListingId, digest) in taken:
            continue
        taken.add((version.skillListingId, digest))
        stamped += 1
        if not dry_run:
            await prisma.models.SkillListingVersion.prisma(tx).update(
                where={"id": version.id}, data={"packageSha256": digest}
            )
    return stamped
