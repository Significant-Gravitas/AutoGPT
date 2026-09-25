"""Marketplace activation is one transaction; personal copies are never targets."""

from uuid import uuid4

from prisma import Prisma

from backend.api.features.store.catalog_release_load import LoadedRelease
from backend.api.features.store.catalog_release_model import (
    ReleaseSnapshot,
    canonical_json,
)
from backend.api.features.store.catalog_release_state import DatabaseState


async def record_release(
    tx: Prisma, release: LoadedRelease, snapshot: ReleaseSnapshot
) -> None:
    await record_snapshot(
        tx,
        release.release_id,
        release.revision,
        release.manifest_sha256,
        release.manifest.model_dump_json(),
        snapshot,
    )


async def record_snapshot(
    tx: Prisma,
    release_id: str,
    revision: str,
    manifest_sha256: str,
    manifest_json: str,
    snapshot: ReleaseSnapshot,
) -> None:
    await tx.execute_raw(
        """INSERT INTO "CatalogueRelease" (id, revision, "manifestSha256", manifest, snapshot)
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)""",
        release_id,
        revision,
        manifest_sha256,
        manifest_json,
        snapshot.model_dump_json(),
    )
    await tx.execute_raw(
        """INSERT INTO "CatalogueReleaseVersion" ("releaseId", "versionId")
        SELECT $1, jsonb_array_elements_text($2::jsonb)""",
        release_id,
        canonical_json(
            sorted(
                {
                    skill.version_id
                    for skill in snapshot.skills.values()
                    if skill.version_id is not None
                }
            )
        ),
    )


async def activate(
    tx: Prisma,
    release_id: str,
    snapshot: ReleaseSnapshot,
    state: DatabaseState,
    *,
    rollback: bool,
) -> None:
    await _guard_external_retirement_links(tx, snapshot, state)
    for slug, skill in snapshot.skills.items():
        changed = await tx.execute_raw(
            """UPDATE "SkillListing" SET "activeVersionId" = $2, "isDeleted" = $3,
                "hasApprovedVersion" = $5, "updatedAt" = now()
            WHERE id = $1 AND slug = $4 AND "owningUserId" IS NULL AND "owningOrgId" IS NULL
                AND ($2::text IS NULL OR EXISTS (SELECT 1 FROM "SkillListingVersion" v
                    WHERE v.id = $2 AND v."skillListingId" = $1))""",
            skill.listing_id,
            skill.version_id,
            skill.retired,
            slug,
            skill.has_approved_version,
        )
        if changed != 1:
            raise ValueError(f"skill {slug} left the approved ownership boundary")
    await _replace_assignments(tx, snapshot)
    await tx.execute_raw(
        """INSERT INTO "CatalogueActivation" (id, "releaseId", "previousReleaseId",
            generation, rollback, snapshot) VALUES ($1, $2, $3, $4, $5, $6::jsonb)""",
        str(uuid4()),
        release_id,
        state.active_release_id,
        state.generation + 1,
        rollback,
        snapshot.model_dump_json(),
    )
    changed = await tx.execute_raw(
        """UPDATE "CatalogueState" SET "activeReleaseId" = $1, generation = generation + 1,
            snapshot = $2::jsonb WHERE id = 'marketplace' AND generation = $3
            AND "activeReleaseId" IS NOT DISTINCT FROM $4""",
        release_id,
        snapshot.model_dump_json(),
        state.generation,
        state.active_release_id,
    )
    if changed != 1:
        raise ValueError("catalogue activation changed concurrently")


async def _replace_assignments(tx: Prisma, snapshot: ReleaseSnapshot) -> None:
    for key, expert in snapshot.experts.items():
        rows = await tx.query_raw(
            """SELECT id FROM "Expert" WHERE id = $1 AND "isTemplate" = true
            AND "ownerUserId" IS NULL AND "organizationId" IS NULL AND "teamId" IS NULL""",
            expert.expert_id,
        )
        if len(rows) != 1:
            raise ValueError(f"expert {key} left the approved ownership boundary")
        changed = await tx.execute_raw(
            """UPDATE "Expert" SET "isArchived" = $2, "updatedAt" = now()
            WHERE id = $1 AND "isTemplate" = true AND "ownerUserId" IS NULL
            AND "organizationId" IS NULL AND "teamId" IS NULL""",
            expert.expert_id,
            expert.is_archived,
        )
        if changed != 1:
            raise ValueError(f"expert {key} ownership changed during activation")
        await tx.execute_raw(
            """DELETE FROM "ExpertSkillListing" es USING "Expert" e
            WHERE es."expertId" = $1 AND e.id = es."expertId" AND e."isTemplate" = true
            AND e."ownerUserId" IS NULL AND e."organizationId" IS NULL AND e."teamId" IS NULL""",
            expert.expert_id,
        )
        listing_ids = [snapshot.skills[slug].listing_id for slug in expert.skills]
        changed = await tx.execute_raw(
            """INSERT INTO "ExpertSkillListing" ("expertId", "skillListingId", position)
            SELECT e.id, s.id, (item.ordinality - 1)::integer
            FROM jsonb_array_elements_text($2::jsonb) WITH ORDINALITY item(id, ordinality)
            JOIN "SkillListing" s ON s.id = item.id
            JOIN "Expert" e ON e.id = $1
            WHERE e."isTemplate" = true AND e."ownerUserId" IS NULL
                AND e."organizationId" IS NULL AND e."teamId" IS NULL
                AND s."owningUserId" IS NULL AND s."owningOrgId" IS NULL""",
            expert.expert_id,
            canonical_json(listing_ids),
        )
        if changed != len(listing_ids):
            raise ValueError(
                f"expert {key} assignments left the approved ownership boundary"
            )


async def _guard_external_retirement_links(
    tx: Prisma, snapshot: ReleaseSnapshot, state: DatabaseState
) -> None:
    rows = await tx.query_raw(
        """SELECT 1 FROM "ExpertSkillListing" WHERE "skillListingId" IN
        (SELECT jsonb_array_elements_text($1::jsonb)) AND "expertId" NOT IN
        (SELECT jsonb_array_elements_text($2::jsonb)) LIMIT 1""",
        canonical_json(
            [
                skill.listing_id
                for slug, skill in snapshot.skills.items()
                if skill.retired
                and (slug not in state.skills or not state.skills[slug].is_deleted)
            ]
        ),
        canonical_json([expert.expert_id for expert in snapshot.experts.values()]),
    )
    if rows:
        raise ValueError(
            "retired skills are still linked to an expert outside the approved scope"
        )
