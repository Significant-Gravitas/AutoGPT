"""Create new package versions without rewriting previously installed versions."""

import base64
import hashlib
from uuid import uuid4

from prisma import Prisma

from backend.api.features.store.catalog_release_load import LoadedPackage, LoadedRelease
from backend.api.features.store.catalog_release_model import (
    Adoption,
    PublishedExpert,
    PublishedSkill,
    ReleaseSnapshot,
    canonical_json,
)
from backend.api.features.store.catalog_release_state import DatabaseState


async def prepare_snapshot(
    tx: Prisma, release: LoadedRelease, state: DatabaseState, adoption: Adoption
) -> ReleaseSnapshot:
    skills = dict(state.previous.skills) if state.previous else {}
    for slug, package in release.packages.items():
        existing = state.skills.get(slug)
        listing_id = existing.id if existing else await _create_listing(tx, slug)
        version_id = await _save_version(tx, listing_id, package)
        skills[slug] = PublishedSkill(listing_id=listing_id, version_id=version_id)
    for slug in release.manifest.retirements:
        existing = state.skills.get(slug)
        if existing is None or existing.active_version_id is None:
            raise ValueError(f"cannot retire nonexistent or unversioned skill {slug}")
        skills[slug] = PublishedSkill(
            listing_id=existing.id, version_id=existing.active_version_id, retired=True
        )
    return ReleaseSnapshot(
        skills=skills,
        experts={
            expert.key: PublishedExpert(
                expert_id=state.experts[expert.key].id,
                skills=expert.skills,
                is_archived=(
                    False
                    if expert.key in adoption.activate_experts
                    else state.experts[expert.key].is_archived
                ),
            )
            for expert in release.manifest.experts
        },
    )


async def _create_listing(tx: Prisma, slug: str) -> str:
    listing_id = str(uuid4())
    await tx.execute_raw(
        """INSERT INTO "SkillListing" (id, slug, "updatedAt") VALUES ($1, $2, now())""",
        listing_id,
        slug,
    )
    return listing_id


async def _save_version(tx: Prisma, listing_id: str, package: LoadedPackage) -> str:
    saved = await tx.query_raw(
        """SELECT v.id FROM "SkillListingVersion" v JOIN "SkillListing" s
        ON s.id = v."skillListingId" WHERE s.id = $1 AND s."owningUserId" IS NULL
        AND s."owningOrgId" IS NULL AND v."organizationId" IS NULL AND v."cataloguePackageSha256" = $2""",
        listing_id,
        package.package_sha256,
    )
    if saved:
        return saved[0]["id"]
    version_id = str(uuid4())
    payload = package.model_dump(exclude={"files"})
    rows = await tx.query_raw(
        """INSERT INTO "SkillListingVersion" (id, version, "updatedAt", name, description,
            body, triggers, categories, "requiredProviders", "sourceSkillSlug", "sourceRepo",
            "sourceUrl", license, "submissionStatus", "skillListingId", "skillMarkdown",
            "cataloguePackageSha256", "catalogueMetadata")
        SELECT $1, COALESCE((SELECT MAX(version) + 1 FROM "SkillListingVersion"
            WHERE "skillListingId" = $2), 1), now(), p.name, p.description, p.body,
            p.triggers, p.categories, p.required_providers, p.slug, p.source_repo,
            p.source_url, p.license, 'APPROVED', $2, p.skill_markdown, p.package_sha256, p.catalogue_metadata
        FROM jsonb_to_record($3::jsonb) AS p(name text, description text, body text,
            triggers text[], categories text[], required_providers text[], slug text,
            source_repo text, source_url text, license text, skill_markdown text,
            package_sha256 text, catalogue_metadata jsonb)
        WHERE EXISTS (SELECT 1 FROM "SkillListing" s WHERE s.id = $2
            AND s."owningUserId" IS NULL AND s."owningOrgId" IS NULL) RETURNING id""",
        version_id,
        listing_id,
        canonical_json(payload),
    )
    if len(rows) != 1:
        raise ValueError(f"skill {package.slug} ownership changed")
    await _save_files(tx, version_id, package)
    return version_id


async def _save_files(tx: Prisma, version_id: str, package: LoadedPackage) -> None:
    payload = [
        {
            "id": str(uuid4()),
            "path": file.relative_path,
            "size": len(file.content),
            "sha256": hashlib.sha256(file.content).hexdigest(),
            "executable": file.is_executable,
            "content": base64.b64encode(file.content).decode("ascii"),
        }
        for file in package.files
    ]
    await tx.execute_raw(
        """INSERT INTO "SkillListingFile" (id, "skillListingVersionId", "relativePath",
            "sizeBytes", sha256, "isExecutable", content)
        SELECT f.id, v.id, f.path, f.size, f.sha256, f.executable, decode(f.content, 'base64')
        FROM jsonb_to_recordset($2::jsonb) AS f(id text, path text, size integer,
            sha256 text, executable boolean, content text)
        JOIN "SkillListingVersion" v ON v.id = $1
        JOIN "SkillListing" s ON s.id = v."skillListingId"
        WHERE s."owningUserId" IS NULL AND s."owningOrgId" IS NULL""",
        version_id,
        canonical_json(payload),
    )
