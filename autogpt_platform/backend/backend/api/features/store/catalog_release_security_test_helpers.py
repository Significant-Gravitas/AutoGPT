"""Real PostgreSQL attacks used by the isolated publisher integration scenario."""

from uuid import uuid4

import pytest

from backend.api.features.store.catalog_release_load import LoadedRelease
from backend.api.features.store.catalog_release_model import Adoption
from backend.api.features.store.catalog_release_service import preview_release
from backend.data import db as database


async def assert_foreign_legacy_references_rejected(
    release: LoadedRelease, adoption: Adoption
) -> None:
    await database.prisma.execute_raw(
        'UPDATE "SkillListing" SET "activeVersionId"=NULL WHERE id=\'catalog-private-skill\''
    )
    await database.prisma.execute_raw(
        "UPDATE \"SkillListing\" SET \"activeVersionId\"='catalog-private-version' WHERE id='catalog-old-skill'"
    )
    with pytest.raises(ValueError, match="another listing or organisation"):
        await preview_release(release, adoption)
    await database.prisma.execute_raw(
        "UPDATE \"SkillListing\" SET \"activeVersionId\"='catalog-old-version' WHERE id='catalog-old-skill'"
    )
    await database.prisma.execute_raw(
        "UPDATE \"SkillListing\" SET \"activeVersionId\"='catalog-private-version' WHERE id='catalog-private-skill'"
    )
    await database.prisma.execute_raw(
        "UPDATE \"SkillListingVersion\" SET body=body WHERE id='catalog-private-version'"
    )
    await database.prisma.execute_raw(
        'INSERT INTO "ExpertSkillListing" ("expertId","skillListingId",position) VALUES (\'catalog-template\',\'catalog-private-skill\',1)'
    )
    with pytest.raises(ValueError, match="outside the approved skill IDs"):
        await preview_release(release, adoption)
    await database.prisma.execute_raw(
        'DELETE FROM "ExpertSkillListing" WHERE "expertId"=\'catalog-template\' AND "skillListingId"=\'catalog-private-skill\''
    )


async def assert_rls_cannot_hide_management(version_id: str) -> None:
    role = f"catalogue_guard_probe_{uuid4().hex}"
    await database.prisma.execute_raw(f"CREATE ROLE {role} NOLOGIN")
    schema = (await database.prisma.query_raw("SELECT current_schema() AS name"))[0][
        "name"
    ]
    assert schema.replace("_", "").isalnum()
    try:
        await database.prisma.execute_raw(f'GRANT USAGE ON SCHEMA "{schema}" TO {role}')
        await database.prisma.execute_raw(
            f'GRANT SELECT, UPDATE ON "SkillListingVersion" TO {role}'
        )
        await database.prisma.execute_raw(f'GRANT SELECT ON "CatalogueState" TO {role}')
        async with database.transaction() as tx:
            await tx.execute_raw(f"SET LOCAL ROLE {role}")
            assert await tx.query_raw('SELECT id FROM "CatalogueState"') == []
            with pytest.raises(Exception, match="immutable"):
                await tx.execute_raw(
                    "UPDATE \"SkillListingVersion\" SET body='bypass' WHERE id=$1",
                    version_id,
                )
    finally:
        await database.prisma.execute_raw(f"DROP OWNED BY {role}")
        await database.prisma.execute_raw(f"DROP ROLE {role}")
