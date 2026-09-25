"""Explicitly provision missing platform templates; never update customer copies.

This is an operator command, not a startup hook. New templates remain archived
until the catalogue publisher explicitly activates them with their assignments.
Existing template fields, routines, preloads and credentials are never updated.
"""

import argparse
import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from prisma import Prisma

from backend.api.features.experts.catalogue_template_models import (
    CatalogueCoordinationState,
    PreloadVersion,
    ProvisionedTemplates,
    TemplateAdoption,
    TemplateDefinition,
    TemplatePreview,
    TemplateRecord,
    template_definitions,
)
from backend.api.features.store.catalog_release_model import canonical_json, digest


async def preview_templates(db: Prisma, adoption: TemplateAdoption) -> TemplatePreview:
    async with db.tx(timeout=timedelta(seconds=60)) as tx:
        preview, _ = await _preview(tx, adoption, lock=False)
        return preview


async def apply_templates(
    db: Prisma, adoption: TemplateAdoption, *, expected_preview_sha256: str
) -> ProvisionedTemplates:
    async with db.tx(timeout=timedelta(seconds=60)) as tx:
        preview, definitions = await _preview(tx, adoption, lock=True)
        if preview.preview_sha256 != expected_preview_sha256:
            raise ValueError(
                "template preview is stale; review a fresh preview before applying"
            )
        result = {
            key: value for key, value in adoption.experts.items() if value is not None
        }
        for key in preview.create_keys:
            expert_id = str(uuid4())
            await _create_template(
                tx, expert_id, definitions[key], preview.preload_versions
            )
            result[key] = expert_id
        return ProvisionedTemplates(
            experts=result, activate_experts=preview.create_keys
        )


async def _preview(
    tx: Prisma, adoption: TemplateAdoption, *, lock: bool
) -> tuple[TemplatePreview, dict[str, TemplateDefinition]]:
    definitions = template_definitions()
    if set(adoption.experts) != set(definitions):
        raise ValueError(
            "adoption must explicitly map every roster key, with null for missing templates"
        )
    state = await _coordination_lock(tx, lock=lock)
    ids = [value for value in adoption.experts.values() if value is not None]
    if lock:
        await _lock_templates(tx, ids)
    rows = await _template_rows(tx, ids)
    await _validate_adoption(tx, adoption, definitions, rows)
    create_keys = sorted(
        key for key, value in adoption.experts.items() if value is None
    )
    slugs = sorted({p.slug for key in create_keys for p in definitions[key].preloads})
    preloads = await _preloads(tx, slugs, lock=lock)
    payload = {
        "state_sha256": digest(
            {
                "catalogue": state.model_dump(),
                "templates": [row.model_dump() for row in rows],
                "preloads": [row.model_dump() for row in preloads],
            }
        ),
        "definitions_sha256": digest(
            {key: definition.model_dump() for key, definition in definitions.items()}
        ),
        "adoption_sha256": digest(adoption.model_dump()),
        "create_keys": create_keys,
        "preserve_keys": sorted(set(adoption.experts) - set(create_keys)),
        "preload_versions": {row.slug: row.version_id for row in preloads},
    }
    return (
        TemplatePreview.model_validate({"preview_sha256": digest(payload), **payload}),
        definitions,
    )


async def _coordination_lock(tx: Prisma, *, lock: bool) -> CatalogueCoordinationState:
    # Same first lock as publication/rollback; serialises cooperating operators.
    rows = await tx.query_raw(
        'SELECT "activeReleaseId", generation FROM "CatalogueState" WHERE id = \'marketplace\''
        + (" FOR UPDATE" if lock else " FOR SHARE")
    )
    if len(rows) != 1:
        raise ValueError("catalogue schema migration / singleton state is missing")
    return CatalogueCoordinationState.model_validate(rows[0])


async def _lock_templates(tx: Prisma, ids: list[str]) -> None:
    # Names are not unique: row locks cannot protect an expected-absent name.
    # This short lock blocks Expert writes without changing unrelated rows.
    await tx.execute_raw('LOCK TABLE "Expert" IN SHARE ROW EXCLUSIVE MODE')
    await tx.query_raw(
        """SELECT id FROM "Expert" WHERE id IN
        (SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY id FOR UPDATE""",
        canonical_json(ids),
    )
    for table in ("ExpertRoutine", "ExpertWorkflow", "ExpertCredential"):
        await tx.query_raw(
            f"""SELECT id FROM "{table}" WHERE "expertId" IN
            (SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY id FOR SHARE""",
            canonical_json(ids),
        )


async def _validate_adoption(
    tx: Prisma,
    adoption: TemplateAdoption,
    definitions: dict[str, TemplateDefinition],
    rows: list[TemplateRecord],
) -> None:
    by_id = {row.id: row for row in rows}
    names = [definition.fields.name for definition in definitions.values()]
    collisions = await tx.query_raw(
        """SELECT id, name FROM "Expert" WHERE "isTemplate" = true
        AND name IN (SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY id""",
        canonical_json(names),
    )
    expected_names = {
        definitions[key].fields.name: record_id
        for key, record_id in adoption.experts.items()
    }
    for collision in collisions:
        if expected_names.get(collision["name"]) != collision["id"]:
            raise ValueError(
                f"unexpected existing template for {collision['name']}; no records changed"
            )
    for key, record_id in adoption.experts.items():
        if record_id is None:
            continue
        row = by_id.get(record_id)
        if (
            row is None
            or not row.isTemplate
            or any(
                value is not None
                for value in (row.ownerUserId, row.organizationId, row.teamId)
            )
        ):
            raise ValueError(
                f"{key} is not the explicitly adopted unowned platform template"
            )
        if row.name != definitions[key].fields.name:
            raise ValueError(f"{key} has an unexpected template name")


async def _template_rows(tx: Prisma, ids: list[str]) -> list[TemplateRecord]:
    rows = await tx.query_raw(
        """SELECT e.id, e.name, e."isTemplate", e."ownerUserId", e."organizationId", e."teamId",
        encode(sha256(convert_to(jsonb_build_object('expert', to_jsonb(e),
            'routines', (SELECT jsonb_agg(to_jsonb(r) ORDER BY r.id)
                FROM "ExpertRoutine" r WHERE r."expertId" = e.id),
            'workflows', (SELECT jsonb_agg(to_jsonb(w) ORDER BY w.id)
                FROM "ExpertWorkflow" w WHERE w."expertId" = e.id),
            'credentials', (SELECT jsonb_agg(to_jsonb(c) ORDER BY c."credentialId")
                FROM "ExpertCredential" c WHERE c."expertId" = e.id),
            'skills', (SELECT jsonb_agg(to_jsonb(s) ORDER BY s.position, s."skillListingId")
                FROM "ExpertSkillListing" s WHERE s."expertId" = e.id)
        )::text, 'UTF8')), 'hex') AS fingerprint FROM "Expert" e
        WHERE e.id IN (SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY e.id""",
        canonical_json(ids),
    )
    return [TemplateRecord.model_validate(row) for row in rows]


async def _preloads(
    tx: Prisma, slugs: list[str], *, lock: bool
) -> list[PreloadVersion]:
    rows = await tx.query_raw(
        """SELECT l.slug, l.id AS listing_id, v.id AS version_id,
        encode(sha256(convert_to(jsonb_build_array(to_jsonb(l), to_jsonb(v), p."userId", p.username)::text,
            'UTF8')), 'hex') AS fingerprint
        FROM "StoreListing" l JOIN "Profile" p ON p."userId" = l."owningUserId"
        JOIN "StoreListingVersion" v ON v.id = l."activeVersionId" AND v."storeListingId" = l.id
        WHERE p.username = 'autogpt' AND NOT l."isDeleted" AND l."hasApprovedVersion"
        AND NOT v."isDeleted" AND v."isAvailable" AND v."submissionStatus" = 'APPROVED'
        AND l.slug IN (SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY l.slug"""
        + (" FOR SHARE OF l, p, v" if lock else ""),
        canonical_json(slugs),
    )
    if len(rows) != len(slugs) or {row["slug"] for row in rows} != set(slugs):
        raise ValueError(
            "missing, ambiguous or unavailable official autogpt preload workflow"
        )
    return [PreloadVersion.model_validate(row) for row in rows]


_PARENT_GUARD = """e.id = $2 AND e."isTemplate" AND e."isArchived"
    AND e."ownerUserId" IS NULL AND e."organizationId" IS NULL AND e."teamId" IS NULL"""


async def _create_template(
    tx: Prisma, expert_id: str, definition: TemplateDefinition, preloads: dict[str, str]
):
    fields = definition.fields.model_dump()
    changed = await tx.execute_raw(
        """INSERT INTO "Expert" (id, "createdAt", "updatedAt", name, role, "jobTitle", tagline,
        "avatarUrl", identity, bio, "voicePreferences", boundaries, categories, "dayOne", "isTemplate", "isArchived")
        SELECT $1, now(), now(), f.name, f.role, f."jobTitle", f.tagline, f."avatarUrl", f.identity,
        f.bio, f."voicePreferences", f.boundaries, ARRAY(SELECT jsonb_array_elements_text(f.categories)),
        f."dayOne", true, true FROM jsonb_to_record($2::jsonb) AS f(name text, role text,
        "jobTitle" text, tagline text, "avatarUrl" text, identity text, bio text,
        "voicePreferences" text, boundaries text, categories jsonb, "dayOne" jsonb)""",
        expert_id,
        canonical_json(fields),
    )
    if changed != 1:
        raise ValueError("failed to create the approved template")
    await _create_routines(tx, expert_id, definition)
    await _create_preloads(tx, expert_id, definition, preloads)


async def _create_routines(
    tx: Prisma, expert_id: str, definition: TemplateDefinition
) -> None:
    for routine in definition.routines:
        changed = await tx.execute_raw(
            """INSERT INTO "ExpertRoutine" (id, "expertId", key, title, prompt, crons, asks,
            "sessionMode", source, "scheduleIds", "grantsCredentials")
            SELECT $1, e.id, r.key, r.title, r.prompt, ARRAY(SELECT jsonb_array_elements_text(r.crons)),
            ARRAY(SELECT jsonb_array_elements_text(r.asks)), r."sessionMode"::"ExpertRoutineSession",
            'TEMPLATE'::"ExpertRoutineSource", ARRAY[]::text[], false
            FROM "Expert" e CROSS JOIN jsonb_to_record($3::jsonb)
            AS r(key text, title text, prompt text, crons jsonb, asks jsonb, "sessionMode" text)
            WHERE """
            + _PARENT_GUARD,
            str(uuid4()),
            expert_id,
            routine.model_dump_json(),
        )
        if changed != 1:
            raise ValueError("routine parent left the platform template boundary")


async def _create_preloads(
    tx: Prisma, expert_id: str, definition: TemplateDefinition, preloads: dict[str, str]
) -> None:
    for preload in definition.preloads:
        changed = await tx.execute_raw(
            """INSERT INTO "ExpertWorkflow" (id, "expertId", "storeListingVersionId", "scheduleCron")
            SELECT $1, e.id, $3, $4 FROM "Expert" e WHERE """
            + _PARENT_GUARD,
            str(uuid4()),
            expert_id,
            preloads[preload.slug],
            preload.cron,
        )
        if changed != 1:
            raise ValueError("preload parent left the platform template boundary")


async def _main(args) -> None:
    adoption = TemplateAdoption.model_validate_json(
        args.adoption.read_text(encoding="utf-8")
    )
    db = Prisma()
    await db.connect()
    try:
        if args.command == "preview":
            result = await preview_templates(db, adoption)
        else:
            previous = TemplatePreview.model_validate_json(
                args.preview.read_text(encoding="utf-8")
            )
            result = await apply_templates(
                db, adoption, expected_preview_sha256=previous.preview_sha256
            )
        args.output.write_text(
            result.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    finally:
        await db.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "apply"))
    parser.add_argument("--adoption", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "apply" and args.preview is None:
        parser.error("apply requires --preview with a reviewed preview file")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
