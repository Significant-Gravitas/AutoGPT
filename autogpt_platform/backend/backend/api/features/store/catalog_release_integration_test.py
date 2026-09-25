"""Run on a fresh disposable PostgreSQL database with the real migration applied.

CATALOGUE_TEST_DATABASE_URL must name a local catalogue_* test database. The
test deliberately refuses reused state and leaves its evidence for inspection.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch
from urllib.parse import urlparse

import pytest

from backend.api.features.store.catalog_release_model import Adoption
from backend.api.features.store.catalog_release_security_test_helpers import (
    assert_foreign_legacy_references_rejected,
    assert_rls_cannot_hide_management,
)
from backend.api.features.store.catalog_release_service import (
    apply_release,
    preview_release,
    preview_rollback,
    rollback_release,
)
from backend.api.features.store.catalog_release_test_helpers import (
    protected_snapshot,
    release,
    seed_personal_and_platform_records,
)
from backend.data import db as database

pytestmark = pytest.mark.catalogue_isolated


@pytest.mark.integration
async def test_postgres_publish_failure_concurrency_rollback_preserves_personal_data():
    url = os.getenv("CATALOGUE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("requires an explicitly selected disposable PostgreSQL database")
    parsed = urlparse(url)
    assert parsed.hostname in {"127.0.0.1", "localhost"}
    assert parsed.path.startswith("/catalogue_")
    assert database.DATABASE_URL == url
    await database.connect()
    try:
        rows = await database.prisma.query_raw(
            'SELECT generation FROM "CatalogueState"'
        )
        assert rows == [{"generation": 0}], "use a fresh disposable test database"
        await _exercise_lifecycle()
    finally:
        await database.disconnect()


async def _exercise_lifecycle():
    adoption = await seed_personal_and_platform_records()
    adoption = adoption.model_copy(
        update={"skills": {**adoption.skills, "catalog-test-baseline-extra": None}}
    )
    personal = await protected_snapshot()
    first = release(1)
    await assert_foreign_legacy_references_rejected(first, adoption)
    preview = await preview_release(first, adoption)
    wrong_ids = adoption.model_copy(
        update={
            "skills": {**adoption.skills, "catalog-test-demo": "catalog-private-skill"}
        }
    )
    with pytest.raises(ValueError, match="adopted record ID"):
        await preview_release(first, wrong_ids)
    await database.prisma.execute_raw(
        "UPDATE \"SkillListingVersion\" SET body = 'Drift' WHERE id = 'catalog-old-version'"
    )
    with pytest.raises(ValueError, match="preview no longer matches"):
        await apply_release(first, adoption, preview)
    preview = await preview_release(first, adoption)
    assert await apply_release(first, adoption, preview) == first.release_id
    assert await protected_snapshot() == personal
    rows = await database.prisma.query_raw(
        """SELECT s.id, v.id AS version_id, v."skillMarkdown", e."isArchived"
        FROM "SkillListing" s JOIN "SkillListingVersion" v ON v.id=s."activeVersionId"
        CROSS JOIN "Expert" e WHERE s.slug='catalog-test-demo' AND e.id='catalog-template' """
    )
    assert rows[0]["id"] == "catalog-old-skill"
    assert (
        rows[0]["skillMarkdown"] == first.packages["catalog-test-demo"].skill_markdown
    )
    assert rows[0]["isArchived"] is False
    first_version = rows[0]["version_id"]
    await _assert_database_guards(first_version)
    await apply_release(first, adoption, preview)
    assert await _generation() == 1
    extra_id = (
        await database.prisma.query_raw(
            "SELECT id FROM \"SkillListing\" WHERE slug='catalog-test-baseline-extra'"
        )
    )[0]["id"]
    adoption = adoption.model_copy(
        update={"skills": {**adoption.skills, "catalog-test-baseline-extra": extra_id}}
    )
    second_adoption = Adoption(
        skills={**adoption.skills, "catalog-test-second": None},
        experts=adoption.experts,
    )
    second = release(2, second=True)
    approved_second = await preview_release(second, second_adoption)
    with patch(
        "backend.api.features.store.catalog_release_service.activate",
        AsyncMock(side_effect=RuntimeError("injected failure")),
    ):
        with pytest.raises(RuntimeError, match="injected failure"):
            await apply_release(second, second_adoption, approved_second)
    assert (
        await database.prisma.query_raw(
            "SELECT id FROM \"SkillListing\" WHERE slug='catalog-test-second'"
        )
        == []
    )
    assert (
        await database.prisma.query_raw(
            'SELECT id FROM "CatalogueRelease" WHERE id=$1', second.release_id
        )
        == []
    )
    assert await _generation() == 1
    assert await protected_snapshot() == personal
    await _assert_reader_lock_blocks_activation(
        second, second_adoption, approved_second
    )
    await apply_release(second, second_adoption, approved_second)
    assert await _generation() == 2
    second_id = (
        await database.prisma.query_raw(
            "SELECT id FROM \"SkillListing\" WHERE slug='catalog-test-second'"
        )
    )[0]["id"]
    all_adopted = Adoption(
        skills={**adoption.skills, "catalog-test-second": second_id},
        experts=adoption.experts,
    )
    third = release(3, retire_first=True)
    approved_third = await preview_release(third, all_adopted)
    competitor = release(4, second=True)
    approved_competitor = await preview_release(competitor, all_adopted)
    results = await asyncio.gather(
        apply_release(third, all_adopted, approved_third),
        apply_release(competitor, all_adopted, approved_competitor),
        return_exceptions=True,
    )
    assert (
        sum(result in {third.release_id, competitor.release_id} for result in results)
        == 1
    )
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert await _generation() == 3
    version_count = await database.prisma.query_raw(
        'SELECT count(*)::integer AS n FROM "SkillListingVersion"'
    )
    rollback = await preview_rollback(first.release_id, all_adopted)
    await rollback_release(first.release_id, all_adopted, rollback)
    assert await _generation() == 4
    assert await protected_snapshot() == personal
    assert (
        await database.prisma.query_raw(
            'SELECT count(*)::integer AS n FROM "SkillListingVersion"'
        )
        == version_count
    )
    restored = await database.prisma.query_raw(
        'SELECT "activeVersionId", "isDeleted" FROM "SkillListing" WHERE slug=\'catalog-test-demo\''
    )
    assert restored == [{"activeVersionId": first_version, "isDeleted": False}]
    assert (
        await database.prisma.query_raw(
            'SELECT "isDeleted" FROM "SkillListing" WHERE slug=\'catalog-test-second\''
        )
    )[0]["isDeleted"]
    links = await database.prisma.query_raw(
        'SELECT "skillListingId" FROM "ExpertSkillListing" WHERE "expertId"=\'catalog-template\' ORDER BY position'
    )
    assert links == [
        {"skillListingId": "catalog-old-skill"},
        {"skillListingId": extra_id},
        {"skillListingId": "catalog-unversioned"},
    ]
    await rollback_release(
        first.release_id,
        all_adopted,
        await preview_rollback(first.release_id, all_adopted),
    )
    assert await _generation() == 4

    backup = await preview_rollback(preview.rollback_release_id, all_adopted)
    assert backup.revision == "database-before-adoption"
    await rollback_release(preview.rollback_release_id, all_adopted, backup)
    assert await protected_snapshot() == personal
    assert await database.prisma.query_raw(
        'SELECT "activeVersionId", "isDeleted", "hasApprovedVersion" FROM "SkillListing" WHERE slug=\'catalog-test-demo\''
    ) == [
        {
            "activeVersionId": "catalog-old-version",
            "isDeleted": False,
            "hasApprovedVersion": True,
        }
    ]
    assert await database.prisma.query_raw(
        'SELECT "activeVersionId", "isDeleted", "hasApprovedVersion" FROM "SkillListing" WHERE slug=\'catalog-test-baseline-extra\''
    ) == [{"activeVersionId": None, "isDeleted": True, "hasApprovedVersion": False}]
    assert await database.prisma.query_raw(
        'SELECT "activeVersionId", "isDeleted", "hasApprovedVersion" FROM "SkillListing" WHERE slug=\'catalog-test-baseline-unversioned\''
    ) == [{"activeVersionId": None, "isDeleted": False, "hasApprovedVersion": False}]
    assert (
        await database.prisma.query_raw(
            'SELECT "isArchived" FROM "Expert" WHERE id=\'catalog-template\''
        )
    )[0]["isArchived"]
    assert await database.prisma.query_raw(
        'SELECT "skillListingId" FROM "ExpertSkillListing" WHERE "expertId"=\'catalog-template\''
    ) == [{"skillListingId": "catalog-old-skill"}]
    assert (
        await database.prisma.query_raw(
            'SELECT count(*)::integer AS n FROM "SkillListingVersion"'
        )
        == version_count
    )
    await rollback_release(
        first.release_id,
        all_adopted,
        await preview_rollback(first.release_id, all_adopted),
    )
    assert await _generation() == 6
    await rollback_release(
        preview.rollback_release_id,
        all_adopted,
        await preview_rollback(preview.rollback_release_id, all_adopted),
    )
    new_release = release(5)
    resumed_adoption = all_adopted.model_copy(update={"activate_experts": ["max"]})
    await apply_release(
        new_release,
        resumed_adoption,
        await preview_release(new_release, resumed_adoption),
    )
    assert await _generation() == 8
    assert await protected_snapshot() == personal


async def _assert_database_guards(version_id):
    rejected = [
        ("UPDATE \"SkillListingVersion\" SET body='tampered' WHERE id=$1", version_id),
        ('DELETE FROM "SkillListingFile" WHERE "skillListingVersionId"=$1', version_id),
        (
            'UPDATE "SkillListingFile" SET "skillListingVersionId"=$1 WHERE id=\'catalog-unmanaged-file\'',
            version_id,
        ),
        (
            'UPDATE "SkillListingVersion" SET "skillListingId"=$1 WHERE id=\'catalog-unmanaged-version\'',
            "catalog-old-skill",
        ),
        ('UPDATE "SkillListing" SET "isDeleted"=true WHERE id=$1', "catalog-old-skill"),
        (
            'UPDATE "SkillListing" SET "owningUserId"=\'catalog-user\' WHERE id=$1',
            "catalog-old-skill",
        ),
        ('DELETE FROM "ExpertSkillListing" WHERE "expertId"=$1', "catalog-template"),
        (
            'UPDATE "Expert" SET "ownerUserId"=\'catalog-user\' WHERE id=$1',
            "catalog-template",
        ),
    ]
    for statement, arg in rejected:
        with pytest.raises(Exception, match="catalogue|managed"):
            await database.prisma.execute_raw(statement, arg)
    await assert_rls_cannot_hide_management(version_id)
    await database.prisma.execute_raw(
        'UPDATE "SkillListingVersion" SET "scannedSha256"=ARRAY[\'cache\'], "updatedAt"=now() WHERE id=$1',
        version_id,
    )
    await database.prisma.execute_raw(
        'UPDATE "SkillListing" SET "installCount"="installCount"+1 WHERE id=\'catalog-old-skill\''
    )
    async with database.transaction() as tx:
        await tx.execute_raw("SET LOCAL autogpt.catalogue_publisher = 'on'")
        with pytest.raises(Exception, match="ownership is immutable"):
            await tx.execute_raw(
                "UPDATE \"SkillListing\" SET \"owningUserId\"='catalog-user' WHERE id='catalog-old-skill'"
            )


async def _assert_reader_lock_blocks_activation(candidate, adoption, approved):
    async with database.transaction() as tx:
        await tx.query_raw(
            "SELECT id FROM \"CatalogueState\" WHERE id='marketplace' FOR SHARE"
        )
        task = asyncio.create_task(apply_release(candidate, adoption, approved))
        await asyncio.sleep(0.1)
        assert not task.done()
    assert await task == candidate.release_id


async def _generation():
    return (await database.prisma.query_raw('SELECT generation FROM "CatalogueState"'))[
        0
    ]["generation"]
