"""The first adoption preserves an explicit, immutable marketplace rollback point."""

from prisma import Prisma

from backend.api.features.store.catalog_release_activate import record_snapshot
from backend.api.features.store.catalog_release_model import (
    PublishedExpert,
    PublishedSkill,
    ReleaseSnapshot,
    canonical_json,
    digest,
)
from backend.api.features.store.catalog_release_state import DatabaseState


def adoption_backup_id(release_id: str, database_target: str) -> str:
    return digest({"adoption_of": release_id, "database_target": database_target})


async def preserve_adoption(
    tx: Prisma, release_id: str, state: DatabaseState, prepared: ReleaseSnapshot
) -> None:
    if state.previous is not None:
        return
    skills = {
        slug: PublishedSkill(
            listing_id=prepared.skills[slug].listing_id,
            version_id=(
                state.skills[slug].active_version_id if slug in state.skills else None
            ),
            retired=state.skills[slug].is_deleted if slug in state.skills else True,
            has_approved_version=(
                state.skills[slug].has_approved_version
                if slug in state.skills
                else False
            ),
        )
        for slug in prepared.skills
    }
    by_id = {skill.listing_id: slug for slug, skill in skills.items()}
    experts = {
        key: PublishedExpert(
            expert_id=row.id,
            skills=[by_id[listing_id] for listing_id in row.skills],
            is_archived=row.is_archived,
        )
        for key, row in state.experts.items()
    }
    descriptor = {
        "kind": "database-before-adoption",
        "adopted_by_release_id": release_id,
    }
    await record_snapshot(
        tx,
        adoption_backup_id(release_id, state.database_target),
        "database-before-adoption",
        digest(descriptor),
        canonical_json(descriptor),
        ReleaseSnapshot(skills=skills, experts=experts),
    )
