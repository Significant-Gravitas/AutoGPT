"""Read and validate the exact marketplace-only write boundary."""

from urllib.parse import parse_qs, urlparse

from prisma import Prisma

from backend.api.features.store.catalog_release_model import (
    Adoption,
    ReleaseManifest,
    ReleaseSnapshot,
    StrictModel,
    canonical_json,
    digest,
)
from backend.data import db as database


class SkillState(StrictModel):
    id: str
    slug: str
    owning_user_id: str | None
    owning_org_id: str | None
    active_version_id: str | None
    active_version_listing_id: str | None
    active_version_organization_id: str | None
    is_deleted: bool
    has_approved_version: bool
    version_fingerprint: str | None
    file_fingerprint: str


class ExpertState(StrictModel):
    id: str
    owner_user_id: str | None
    organization_id: str | None
    team_id: str | None
    is_template: bool
    is_archived: bool
    skills: list[str]


class DatabaseState(StrictModel):
    database_target: str
    active_release_id: str | None
    generation: int
    skills: dict[str, SkillState]
    experts: dict[str, ExpertState]
    previous: ReleaseSnapshot | None = None

    def fingerprint(self) -> str:
        return digest(self.model_dump(exclude={"previous"}))


async def read_state(tx: Prisma, adoption: Adoption, *, lock: bool) -> DatabaseState:
    row = await tx.query_raw(
        'SELECT "activeReleaseId", generation, snapshot::text AS snapshot FROM "CatalogueState" '
        "WHERE id = 'marketplace'" + (" FOR UPDATE" if lock else " FOR SHARE")
    )
    if len(row) != 1:
        raise ValueError("catalogue schema migration / singleton state is missing")
    if lock:
        await _lock_parents(tx, adoption)
    skills = await _read_skills(tx, list(adoption.skills))
    experts = await _read_experts(tx, adoption.experts)
    active = row[0]["activeReleaseId"]
    previous = (
        ReleaseSnapshot.model_validate_json(row[0]["snapshot"])
        if row[0]["snapshot"]
        else None
    )
    if bool(active) != bool(previous):
        raise ValueError("catalogue state pointer and snapshot disagree")
    return DatabaseState(
        database_target=_database_target(),
        active_release_id=active,
        generation=row[0]["generation"],
        skills=skills,
        experts=experts,
        previous=previous,
    )


def validate_boundary(
    state: DatabaseState,
    adoption: Adoption,
    manifest: ReleaseManifest,
    *,
    replay_release_id: str | None = None,
) -> None:
    validate_scope(
        state,
        adoption,
        {package.slug for package in manifest.packages},
        set(manifest.retirements),
        {expert.key for expert in manifest.experts},
        replay_release_id=replay_release_id,
    )
    for slug in manifest.retirements:
        row = state.skills.get(slug)
        if row is None or row.active_version_id is None:
            raise ValueError(f"cannot retire nonexistent or unversioned skill {slug}")


def validate_scope(
    state: DatabaseState,
    adoption: Adoption,
    active_slugs: set[str],
    retirements: set[str],
    expert_keys: set[str],
    *,
    replay_release_id: str | None = None,
) -> None:
    expected = active_slugs | retirements
    if state.previous:
        expected |= state.previous.skills.keys()
    if set(adoption.skills) != expected:
        raise ValueError(
            "adoption must explicitly map every active, retired and managed skill"
        )
    if set(adoption.experts) != expert_keys:
        raise ValueError("adoption must explicitly map every manifest expert")
    for slug, approved_id in adoption.skills.items():
        row = state.skills.get(slug)
        if approved_id is None and row is not None:
            previous = state.previous.skills.get(slug) if state.previous else None
            if (
                replay_release_id != state.active_release_id
                or previous is None
                or previous.listing_id != row.id
            ):
                raise ValueError(
                    f"skill {slug} was expected absent, but a listing exists"
                )
        if approved_id is not None and (row is None or row.id != approved_id):
            raise ValueError(f"skill {slug} does not match its adopted record ID")
        if row and (row.owning_user_id is not None or row.owning_org_id is not None):
            raise ValueError(f"skill {slug} belongs to a user or organisation")
        if (
            row
            and row.active_version_id is not None
            and (
                row.active_version_listing_id != row.id
                or row.active_version_organization_id is not None
            )
        ):
            raise ValueError(
                f"skill {slug} active version belongs to another listing or organisation"
            )
    for key, approved_id in adoption.experts.items():
        row = state.experts.get(key)
        if row is None or row.id != approved_id:
            raise ValueError(f"expert {key} does not match its adopted record ID")
        if not row.is_template or any(
            value is not None
            for value in (row.owner_user_id, row.organization_id, row.team_id)
        ):
            raise ValueError(f"expert {key} is not an unowned platform template")
    _validate_previous(state, adoption, active_slugs, retirements)
    if state.previous is None:
        adopted_ids = {row.id for row in state.skills.values()}
        if any(set(row.skills) - adopted_ids for row in state.experts.values()):
            raise ValueError(
                "pre-adoption expert assignments extend outside the approved skill IDs"
            )


def _database_target() -> str:
    parsed = urlparse(database.DATABASE_URL)
    schema = parse_qs(parsed.query).get("schema", ["public"])[0]
    return f"{parsed.hostname}:{parsed.port or 5432}{parsed.path}?schema={schema}"


def _validate_previous(state, adoption, active_slugs, retirements) -> None:
    previous = state.previous
    if previous is None:
        return
    for slug, saved in previous.skills.items():
        row = state.skills.get(slug)
        if (
            row is None
            or row.id != saved.listing_id
            or row.active_version_id != saved.version_id
        ):
            raise ValueError(f"managed skill {slug} has drifted")
        if (
            row.is_deleted != saved.retired
            or row.has_approved_version != saved.has_approved_version
        ):
            raise ValueError(f"managed skill {slug} visibility has drifted")
        if not saved.retired and slug not in active_slugs and slug not in retirements:
            raise ValueError(f"skill {slug} disappeared without explicit retirement")
    for key, saved in previous.experts.items():
        row = state.experts.get(key)
        expected = [previous.skills[slug].listing_id for slug in saved.skills]
        if (
            adoption.experts.get(key) != saved.expert_id
            or row is None
            or row.skills != expected
        ):
            raise ValueError(
                f"managed expert {key} assignments have drifted or disappeared"
            )
        if row.is_archived != saved.is_archived:
            raise ValueError(f"managed expert {key} visibility has drifted")


async def read_release_snapshot(tx: Prisma, release_id: str) -> ReleaseSnapshot:
    rows = await tx.query_raw(
        'SELECT snapshot::text AS snapshot FROM "CatalogueRelease" WHERE id = $1',
        release_id,
    )
    if len(rows) != 1:
        raise ValueError("recorded catalogue release does not exist")
    return ReleaseSnapshot.model_validate_json(rows[0]["snapshot"])


async def _lock_parents(tx: Prisma, adoption: Adoption) -> None:
    await tx.query_raw(
        'SELECT id FROM "SkillListing" WHERE slug IN '
        "(SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY id FOR UPDATE",
        canonical_json(list(adoption.skills)),
    )
    await tx.query_raw(
        'SELECT id FROM "Expert" WHERE id IN '
        "(SELECT jsonb_array_elements_text($1::jsonb)) ORDER BY id FOR UPDATE",
        canonical_json(list(adoption.experts.values())),
    )


async def _read_skills(tx: Prisma, slugs: list[str]) -> dict[str, SkillState]:
    rows = await tx.query_raw(
        """SELECT jsonb_build_object(
            'id', s.id, 'slug', s.slug, 'owning_user_id', s."owningUserId",
            'owning_org_id', s."owningOrgId", 'active_version_id', s."activeVersionId",
            'active_version_listing_id', v."skillListingId", 'active_version_organization_id', v."organizationId",
            'is_deleted', s."isDeleted", 'has_approved_version', s."hasApprovedVersion",
            'version_fingerprint', encode(sha256(convert_to((to_jsonb(v)
                - 'scannedSha256' - 'updatedAt' - 'createdAt')::text, 'UTF8')), 'hex'),
            'file_fingerprint', encode(sha256(convert_to(COALESCE((SELECT jsonb_agg(
                jsonb_build_array(f."relativePath", f."sizeBytes", f.sha256,
                    f."isExecutable", encode(sha256(f.content), 'hex'))
                ORDER BY f."relativePath")::text FROM "SkillListingFile" f
                WHERE f."skillListingVersionId" = v.id), '[]'), 'UTF8')), 'hex')
        )::text AS payload FROM "SkillListing" s
        LEFT JOIN "SkillListingVersion" v ON v.id = s."activeVersionId"
        WHERE s.slug IN (SELECT jsonb_array_elements_text($1::jsonb))""",
        canonical_json(slugs),
    )
    parsed = [SkillState.model_validate_json(row["payload"]) for row in rows]
    return {row.slug: row for row in parsed}


async def _read_experts(tx: Prisma, experts: dict[str, str]) -> dict[str, ExpertState]:
    rows = await tx.query_raw(
        """SELECT jsonb_build_object('id', e.id, 'owner_user_id', e."ownerUserId",
            'organization_id', e."organizationId", 'team_id', e."teamId",
            'is_template', e."isTemplate", 'is_archived', e."isArchived", 'skills', COALESCE((SELECT jsonb_agg(
                es."skillListingId" ORDER BY es.position, es."skillListingId")
                FROM "ExpertSkillListing" es WHERE es."expertId" = e.id), '[]'::jsonb)
        )::text AS payload FROM "Expert" e
        WHERE e.id IN (SELECT jsonb_array_elements_text($1::jsonb))""",
        canonical_json(list(experts.values())),
    )
    by_id = {
        row.id: row
        for row in [ExpertState.model_validate_json(r["payload"]) for r in rows]
    }
    return {
        key: by_id[record_id]
        for key, record_id in experts.items()
        if record_id in by_id
    }
