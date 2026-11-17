"""The marketplace's slug → active version map is built from the few columns
it needs; whole version rows are read only for versions with no stored hash."""

import datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.enums
import prisma.models
import pytest

from . import skill_db, skill_model


def _row(slug: str, **overrides) -> skill_db._ActiveVersionRow:
    fields = {
        "listing_id": f"listing-{slug}",
        "slug": slug,
        "listing_deleted": False,
        "has_approved": True,
        "version_id": f"version-{slug}",
        "package_sha256": f"sha-{slug}",
        "available": True,
        "version_deleted": False,
        "submission_status": "APPROVED",
    } | overrides
    return skill_db._ActiveVersionRow(**fields)


def _legacy_version(version_id: str) -> prisma.models.SkillListingVersion:
    now = datetime.datetime.now(datetime.timezone.utc)
    return prisma.models.SkillListingVersion(
        id=version_id,
        version=2,
        createdAt=now,
        updatedAt=now,
        name="Old hand",
        description="Old hand description",
        body="# Old hand\n\nStill useful.\n",
        triggers=["old"],
        categories=["content"],
        requiredProviders=[],
        scannedSha256=[],
        isDeleted=False,
        isAvailable=True,
        submissionStatus=prisma.enums.SubmissionStatus.APPROVED,
        skillListingId="listing-old-hand",
    )


@pytest.fixture
def database(mocker):
    """The map's three reads: its query, whole version rows, file hashes."""
    reads = MagicMock(
        rows=AsyncMock(return_value=[]),
        versions=AsyncMock(return_value=[]),
        file_hashes=AsyncMock(return_value={}),
        listings=AsyncMock(return_value=[]),
    )
    mocker.patch.object(skill_db, "query_raw_with_schema", reads.rows)
    mocker.patch.object(skill_db, "_file_hashes", reads.file_hashes)
    mocker.patch.object(
        prisma.models.SkillListingVersion,
        "prisma",
        return_value=MagicMock(find_many=reads.versions),
    )
    mocker.patch.object(
        prisma.models.SkillListing,
        "prisma",
        return_value=MagicMock(find_many=reads.listings),
    )
    return reads


@pytest.mark.asyncio
async def test_whole_rows_are_read_only_for_versions_with_no_stored_hash(database):
    legacy = _legacy_version("version-old-hand")
    database.rows.return_value = [
        _row("cold-email"),
        _row("old-hand", package_sha256=None),
    ]
    database.versions.return_value = [legacy]
    database.file_hashes.return_value = {
        "version-old-hand": [("references/notes.md", "f" * 64, False)]
    }

    versions = await skill_db._load_active_versions()

    database.listings.assert_not_awaited()
    assert database.versions.await_args.kwargs == {
        "where": {"id": {"in": ["version-old-hand"]}}
    }
    database.file_hashes.assert_awaited_once_with(["version-old-hand"])
    assert versions["cold-email"] == skill_model.ActiveSkillVersion(
        listing_id="listing-cold-email",
        version_id="version-cold-email",
        package_sha256="sha-cold-email",
        retired=False,
    )
    assert versions["old-hand"].package_sha256 == skill_model.package_sha256_of(
        skill_model.legacy_skill_markdown(legacy, "old-hand"),
        [("references/notes.md", "f" * 64, False)],
    )


@pytest.mark.asyncio
async def test_a_map_of_hashed_versions_reads_no_whole_rows(database):
    database.rows.return_value = [_row("cold-email"), _row("brand-voice-guide")]

    versions = await skill_db._load_active_versions()

    assert set(versions) == {"cold-email", "brand-voice-guide"}
    database.versions.assert_not_awaited()
    database.file_hashes.assert_not_awaited()
    database.listings.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "withdrawn",
    [
        {"listing_deleted": True},
        {"has_approved": False},
        {"available": False},
        {"version_deleted": True},
        {"submission_status": "PENDING"},
    ],
    ids=lambda w: next(iter(w)),
)
async def test_a_withdrawn_listing_or_version_is_retired(database, withdrawn):
    database.rows.return_value = [_row("live"), _row("withdrawn", **withdrawn)]

    versions = await skill_db._load_active_versions()

    assert not versions["live"].retired
    assert versions["withdrawn"].retired
