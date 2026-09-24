"""Database access for marketplace skill listings.

Browse is rooted at :class:`SkillListingVersion` rather than the listing because
every filter and the ordering read version columns; ``ActiveFor`` constrains the
row to the listing's live version. Install count is displayed but is not a sort
key — a catalogue this size gains nothing from it, and it lives on the listing
because installs accumulate across versions. It counts distinct installs per
owner (personal library or expert): a re-install overwrites that copy and is
not counted again.
"""

import logging

import prisma.enums
import prisma.models
import prisma.types

from backend.copilot.tools.skills import (
    SKILL_ORIGIN_MARKETPLACE,
    SkillFile,
    SkillWrite,
    StoredSkill,
    store_user_skills,
)
from backend.data.db import query_raw_with_schema
from backend.util.exceptions import NotFoundError
from backend.util.models import Pagination

from . import skill_model
from .categories import category_filter_values

logger = logging.getLogger(__name__)

_LISTING_INCLUDE: prisma.types.SkillListingInclude = {
    "ActiveVersion": True,
    "CreatorProfile": True,
}


async def get_marketplace_skills(
    *,
    category: str | None = None,
    search_query: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> skill_model.MarketplaceSkillsResponse:
    where = _live_version_where(category, search_query)
    total = await prisma.models.SkillListingVersion.prisma().count(where=where)
    versions = await prisma.models.SkillListingVersion.prisma().find_many(
        where=where,
        include={"ActiveFor": {"include": _LISTING_INCLUDE}},
        # Seeded versions share a timestamp, and an undefined order between
        # pages then drops or repeats a row; `id` is the uuid primary key.
        order=[{"updatedAt": "desc"}, {"id": "desc"}],
        skip=(page - 1) * page_size,
        take=page_size,
    )
    listings = [v.ActiveFor for v in versions if v.ActiveFor is not None]
    return skill_model.MarketplaceSkillsResponse(
        skills=[skill_model.MarketplaceSkill.from_db(listing) for listing in listings],
        pagination=Pagination(
            total_items=total,
            total_pages=(total + page_size - 1) // page_size,
            current_page=page,
            page_size=page_size,
        ),
    )


async def get_marketplace_skill(slug: str) -> skill_model.MarketplaceSkillDetails:
    listing = await _find_live_listing(slug)
    version = skill_model.active_version(listing)
    return skill_model.MarketplaceSkillDetails.from_db(
        listing, files=await _list_version_file_meta(version.id)
    )


async def find_readable_version(
    slug: str, *, version_id: str | None, user_id: str | None
) -> prisma.models.SkillListingVersion:
    """The version of *slug* whose package *user_id* may read.

    Without a version id the live one, which is public like the listing it
    serves. With one, the live version stays public and any other — a pending
    submission — is the submitter's alone, so a stranger cannot read a package
    the marketplace has not approved. Admins come through the admin router,
    which carries its own check.
    """
    if version_id is None:
        return skill_model.active_version(await _find_live_listing(slug))

    version = await prisma.models.SkillListingVersion.prisma().find_unique(
        where={"id": version_id}, include={"SkillListing": True}
    )
    listing = version.SkillListing if version is not None else None
    if version is None or listing is None or listing.slug != slug:
        raise NotFoundError(f"Skill '{slug}' has no version {version_id}")
    if listing.activeVersionId == version.id:
        # Through the same query the anonymous path uses, so a listing taken
        # down keeps its pointer without keeping its visibility.
        await _find_live_listing(slug)
        return version
    if user_id is None or listing.owningUserId != user_id:
        raise NotFoundError(f"Skill '{slug}' has no version {version_id}")
    return version


async def get_package_file_meta(
    skill_listing_version_id: str, relative_path: str
) -> skill_model.SkillPackageFile | None:
    """One published file's metadata, bytes excluded.

    The size cap is decided on this, so refusing an oversized file never
    reads it.
    """
    rows = await query_raw_with_schema(
        _FILE_META_SELECT + 'WHERE "skillListingVersionId" = $1 '
        'AND "relativePath" = $2',
        skill_listing_version_id,
        relative_path,
    )
    return _to_file_meta(rows[0]) if rows else None


async def read_package_file_bytes(
    skill_listing_version_id: str, relative_path: str
) -> bytes | None:
    """The file's stored bytes, once the caps have let it through."""
    row = await prisma.models.SkillListingFile.prisma().find_unique(
        where={
            "skillListingVersionId_relativePath": {
                "skillListingVersionId": skill_listing_version_id,
                "relativePath": relative_path,
            }
        }
    )
    return row.content.decode() if row is not None else None


async def file_meta_by_version(
    version_ids: list[str],
) -> dict[str, list[skill_model.SkillPackageFile]]:
    """Every version's file list in one query, keyed by version id."""
    if not version_ids:
        return {}
    placeholders = ", ".join(f"${i + 1}" for i in range(len(version_ids)))
    rows = await query_raw_with_schema(
        _FILE_META_SELECT
        + 'WHERE "skillListingVersionId" IN ('
        + placeholders
        + ') ORDER BY "relativePath"',
        *version_ids,
    )
    by_version: dict[str, list[skill_model.SkillPackageFile]] = {}
    for row in rows:
        by_version.setdefault(row["skillListingVersionId"], []).append(
            _to_file_meta(row)
        )
    return by_version


async def get_live_skills(
    listing_ids: list[str],
) -> dict[str, skill_model.MarketplaceSkill]:
    """The live listings among *listing_ids*, keyed by id, in one query."""
    if not listing_ids:
        return {}
    listings = await prisma.models.SkillListing.prisma().find_many(
        where=_live_listing_where({"id": {"in": listing_ids}}),
        include=_LISTING_INCLUDE,
    )
    return {
        listing.id: skill_model.MarketplaceSkill.from_db(listing)
        for listing in listings
    }


async def install_marketplace_skill(
    user_id: str, slug: str, *, expert_id: str | None = None
) -> skill_model.InstalledSkill:
    """Copy a listing's whole package into *expert_id*'s skill folder, or the
    caller's own library when ``None``.

    The listing's slug becomes the installed skill's name, so an install is
    idempotent and a re-install picks up a newer approved version. The whole
    package is passed, so a file the new version dropped is removed too.
    """
    [outcome] = await install_marketplace_skills(user_id, [slug], expert_id=expert_id)
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


async def install_marketplace_skills(
    user_id: str, slugs: list[str], *, expert_id: str | None = None
) -> list[skill_model.InstalledSkill | Exception]:
    """:func:`install_marketplace_skill` for several listings: one listing
    query, one file query and one locked write, with each slug's outcome
    returned in order (``NotFoundError`` for one that is not live)."""
    listings = {
        listing.slug: listing
        for listing in await prisma.models.SkillListing.prisma().find_many(
            where=_live_listing_where({"slug": {"in": slugs}}),
            include=_LISTING_INCLUDE,
        )
    }
    live = [listings[slug] for slug in slugs if slug in listings]
    files = await _read_versions_files(
        [skill_model.active_version(listing).id for listing in live]
    )
    writes: list[SkillWrite] = []
    for listing in live:
        active = skill_model.active_version(listing)
        writes.append(
            SkillWrite(
                name=listing.slug,
                description=active.description,
                body=active.body,
                triggers=list(active.triggers),
                version=str(active.version),
                extra={
                    key: value
                    for key, value in (
                        ("license", active.license),
                        ("source", active.sourceRepo),
                        ("source_url", active.sourceUrl),
                    )
                    if value is not None
                },
                # `[]`, never `None` — which means "leave the folder alone"
                # and would keep a sibling only the previous version had.
                files=files.get(active.id, []),
                scanned_checksums=frozenset(active.scannedSha256),
            )
        )
    stored = (
        dict(
            zip(
                (listing.slug for listing in live),
                await store_user_skills(
                    user_id,
                    writes,
                    expert_id=expert_id,
                    origin=SKILL_ORIGIN_MARKETPLACE,
                ),
            )
        )
        if writes
        else {}
    )
    # A re-install overwrites the existing copy, so counting it again would
    # report installs rather than installers.
    new_ids = [
        listings[slug].id
        for slug, outcome in stored.items()
        if isinstance(outcome, StoredSkill) and outcome.is_new
    ]
    if new_ids:
        await prisma.models.SkillListing.prisma().update_many(
            where={"id": {"in": new_ids}}, data={"installCount": {"increment": 1}}
        )
    await _record_scanned(live, stored)
    outcomes: list[skill_model.InstalledSkill | Exception] = []
    for slug in slugs:
        outcome = stored.get(slug)
        if outcome is None:
            outcomes.append(NotFoundError(f"Skill '{slug}' not found"))
        elif isinstance(outcome, Exception):
            outcomes.append(outcome)
        else:
            outcomes.append(
                skill_model.InstalledSkill(
                    name=slug,
                    required_providers=list(
                        skill_model.active_version(listings[slug]).requiredProviders
                    ),
                )
            )
    return outcomes


async def _record_scanned(
    live: list[prisma.models.SkillListing],
    stored: dict[str, StoredSkill | Exception],
) -> None:
    """Remember the bytes each install wrote past the scan, so the next
    install of the same version writing the same bytes need not scan them."""
    for listing in live:
        outcome = stored.get(listing.slug)
        active = skill_model.active_version(listing)
        if not isinstance(outcome, StoredSkill) or outcome.checksums <= set(
            active.scannedSha256
        ):
            continue
        try:
            await prisma.models.SkillListingVersion.prisma().update(
                where={"id": active.id},
                data={
                    "scannedSha256": sorted(
                        outcome.checksums | set(active.scannedSha256)
                    )
                },
            )
        except Exception:
            # Only a cache of scan results: failing to record costs a rescan
            # next time, never this install.
            logger.warning(
                f"Could not record scanned checksums for skill '{listing.slug}'",
                exc_info=True,
            )


async def _read_versions_files(
    version_ids: list[str],
) -> dict[str, list[SkillFile]]:
    """Each version's published files, contents included, in one query."""
    if not version_ids:
        return {}
    rows = await prisma.models.SkillListingFile.prisma().find_many(
        where={"skillListingVersionId": {"in": version_ids}},
        order={"relativePath": "asc"},
    )
    by_version: dict[str, list[SkillFile]] = {}
    for row in rows:
        by_version.setdefault(row.skillListingVersionId, []).append(
            SkillFile(
                relative_path=row.relativePath,
                content=row.content.decode(),
                is_executable=row.isExecutable,
            )
        )
    return by_version


async def _list_version_file_meta(
    skill_listing_version_id: str,
) -> list[skill_model.SkillPackageFile]:
    return (await file_meta_by_version([skill_listing_version_id])).get(
        skill_listing_version_id, []
    )


# Raw because prisma-client-py always selects every column, and a page that
# pulled `content` would carry the whole package per request.
_FILE_META_SELECT = (
    'SELECT "skillListingVersionId", "relativePath", "sizeBytes", "mimeType", '
    '"isExecutable" FROM {schema_prefix}"SkillListingFile" '
)


def _to_file_meta(row: dict) -> skill_model.SkillPackageFile:
    return skill_model.SkillPackageFile(
        path=row["relativePath"],
        size_bytes=row["sizeBytes"],
        mime_type=row["mimeType"],
        is_executable=row["isExecutable"],
    )


async def _find_live_listing(slug: str) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma().find_first(
        where=_live_listing_where({"slug": slug}), include=_LISTING_INCLUDE
    )
    if listing is None:
        raise NotFoundError(f"Skill '{slug}' not found")
    return listing


def _live_listing_where(
    match: prisma.types.SkillListingWhereInput,
) -> prisma.types.SkillListingWhereInput:
    return {
        **match,
        "isDeleted": False,
        "hasApprovedVersion": True,
        "ActiveVersion": {
            "is": {
                "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
                "isAvailable": True,
                "isDeleted": False,
            }
        },
    }


def _live_version_where(
    category: str | None, search_query: str | None
) -> prisma.types.SkillListingVersionWhereInput:
    where: prisma.types.SkillListingVersionWhereInput = {
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        "isAvailable": True,
        "isDeleted": False,
        "ActiveFor": {"is": {"isDeleted": False, "hasApprovedVersion": True}},
    }
    if category_values := category_filter_values(category):
        where["categories"] = {"has_some": category_values}
    if search_query and search_query.strip():
        needle = search_query.strip()
        where["OR"] = [
            {"name": {"contains": needle, "mode": "insensitive"}},
            {"description": {"contains": needle, "mode": "insensitive"}},
        ]
    return where
