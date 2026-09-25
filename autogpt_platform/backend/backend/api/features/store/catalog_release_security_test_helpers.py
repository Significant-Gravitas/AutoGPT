"""Real PostgreSQL attacks used by the isolated publisher integration scenario."""

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
    async with database.transaction() as tx:
        await tx.query_raw(
            "SELECT set_config('autogpt.catalogue_test_version', $1, true)", version_id
        )
        await tx.execute_raw(
            """DO $$
            DECLARE
                probe_role text := 'catalogue_guard_probe_' || replace(gen_random_uuid()::text, '-', '');
                original_role text := current_user;
                blocked boolean := false;
                failure text;
            BEGIN
                EXECUTE format('CREATE ROLE %I NOLOGIN', probe_role);
                EXECUTE format('GRANT USAGE ON SCHEMA %I TO %I', current_schema(), probe_role);
                EXECUTE format('GRANT SELECT, UPDATE ON "SkillListingVersion" TO %I', probe_role);
                EXECUTE format('GRANT SELECT ON "CatalogueState" TO %I', probe_role);
                EXECUTE format('SET LOCAL ROLE %I', probe_role);
                IF EXISTS (SELECT 1 FROM "CatalogueState") THEN
                    RAISE EXCEPTION 'RLS exposed catalogue management to the probe role';
                END IF;
                BEGIN
                    UPDATE "SkillListingVersion" SET body = 'bypass'
                    WHERE id = current_setting('autogpt.catalogue_test_version');
                EXCEPTION WHEN raise_exception THEN
                    GET STACKED DIAGNOSTICS failure = MESSAGE_TEXT;
                    blocked := failure LIKE '%catalogue versions are immutable%';
                END;
                EXECUTE format('SET LOCAL ROLE %I', original_role);
                IF NOT blocked THEN
                    RAISE EXCEPTION 'RLS hid catalogue management from the immutable-version guard';
                END IF;
                EXECUTE format('DROP OWNED BY %I', probe_role);
                EXECUTE format('DROP ROLE %I', probe_role);
            END $$"""
        )
