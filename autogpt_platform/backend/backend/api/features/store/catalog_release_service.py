"""Preview, publish and roll back an explicitly approved catalogue release."""

from datetime import timedelta

from backend.api.features.store.catalog_release_activate import activate, record_release
from backend.api.features.store.catalog_release_backup import (
    adoption_backup_id,
    preserve_adoption,
)
from backend.api.features.store.catalog_release_load import LoadedRelease
from backend.api.features.store.catalog_release_model import (
    Adoption,
    Preview,
    ReleaseSnapshot,
    digest,
)
from backend.api.features.store.catalog_release_state import (
    DatabaseState,
    read_release_snapshot,
    read_state,
    validate_boundary,
    validate_scope,
)
from backend.api.features.store.catalog_release_versions import prepare_snapshot
from backend.data import db as database

TIMEOUT = timedelta(minutes=10)


async def preview_release(release: LoadedRelease, adoption: Adoption) -> Preview:
    async with database.transaction(timeout=TIMEOUT) as tx:
        state = await read_state(tx, adoption, lock=False)
        validate_boundary(
            state, adoption, release.manifest, replay_release_id=release.release_id
        )
        return _preview(release, adoption, state)


async def apply_release(
    release: LoadedRelease, adoption: Adoption, approved: Preview
) -> str:
    async with database.transaction(timeout=TIMEOUT) as tx:
        state = await read_state(tx, adoption, lock=True)
        validate_boundary(
            state, adoption, release.manifest, replay_release_id=release.release_id
        )
        actual = _preview(release, adoption, state)
        if state.active_release_id == release.release_id:
            _require_release_identity(actual, approved)
            return release.release_id
        _require_approval(actual, approved)
        existing = await tx.query_raw(
            'SELECT id FROM "CatalogueRelease" WHERE id = $1', release.release_id
        )
        if existing:
            raise ValueError(
                "release already exists; use the explicit rollback preview and command"
            )
        await tx.execute_raw("SET LOCAL autogpt.catalogue_publisher = 'on'")
        snapshot = await prepare_snapshot(tx, release, state, adoption)
        await preserve_adoption(tx, release.release_id, state, snapshot)
        await record_release(tx, release, snapshot)
        await activate(tx, release.release_id, snapshot, state, rollback=False)
    return release.release_id


async def preview_rollback(release_id: str, adoption: Adoption) -> Preview:
    async with database.transaction(timeout=TIMEOUT) as tx:
        state = await read_state(tx, adoption, lock=False)
        revision, snapshot = await _rollback_target(tx, release_id, state)
        _validate_rollback_scope(state, adoption, snapshot)
        _validate_rollback_ids(snapshot, adoption)
        return _snapshot_preview(release_id, revision, snapshot, adoption, state)


async def rollback_release(
    release_id: str, adoption: Adoption, approved: Preview
) -> str:
    async with database.transaction(timeout=TIMEOUT) as tx:
        state = await read_state(tx, adoption, lock=True)
        revision, snapshot = await _rollback_target(tx, release_id, state)
        _validate_rollback_scope(state, adoption, snapshot)
        _validate_rollback_ids(snapshot, adoption)
        actual = _snapshot_preview(release_id, revision, snapshot, adoption, state)
        if state.active_release_id == release_id:
            _require_release_identity(actual, approved)
            return release_id
        _require_approval(actual, approved)
        await tx.execute_raw("SET LOCAL autogpt.catalogue_publisher = 'on'")
        await activate(tx, release_id, snapshot, state, rollback=True)
    return release_id


def _preview(
    release: LoadedRelease, adoption: Adoption, state: DatabaseState
) -> Preview:
    return Preview(
        database_target=state.database_target,
        rollback_release_id=state.active_release_id
        or adoption_backup_id(release.release_id, state.database_target),
        release_id=release.release_id,
        revision=release.revision,
        previous_release_id=state.active_release_id,
        generation=state.generation,
        state_sha256=state.fingerprint(),
        adoption_sha256=digest(adoption.model_dump()),
        create_skills=sorted(
            package.slug
            for package in release.manifest.packages
            if package.slug not in state.skills
        ),
        update_skills=sorted(
            package.slug
            for package in release.manifest.packages
            if package.slug in state.skills
        ),
        retire_skills=sorted(release.manifest.retirements),
        expert_keys=sorted(adoption.experts),
        activate_experts=sorted(adoption.activate_experts),
    )


def _require_approval(actual: Preview, approved: Preview) -> None:
    if actual != approved:
        raise ValueError(
            "approved preview no longer matches; regenerate and review the preview"
        )


def _require_release_identity(actual: Preview, approved: Preview) -> None:
    if (
        actual.database_target,
        actual.release_id,
        actual.revision,
        actual.adoption_sha256,
    ) != (
        approved.database_target,
        approved.release_id,
        approved.revision,
        approved.adoption_sha256,
    ):
        raise ValueError("approved preview identifies a different release or adoption")


async def _rollback_target(
    tx, release_id: str, state: DatabaseState
) -> tuple[str, ReleaseSnapshot]:
    if state.previous is None:
        raise ValueError("there is no adopted release to roll back")
    rows = await tx.query_raw(
        'SELECT revision FROM "CatalogueRelease" WHERE id = $1',
        release_id,
    )
    if len(rows) != 1:
        raise ValueError("rollback target is not a previously published release")
    target = await read_release_snapshot(tx, release_id)
    if set(target.experts) != set(state.previous.experts):
        raise ValueError(
            "rollback across expert provisioning requires a separate reviewed template migration"
        )
    skills = dict(target.skills)
    for slug, skill in state.previous.skills.items():
        if slug not in skills:
            skills[slug] = skill.model_copy(update={"retired": True})
    return rows[0]["revision"], ReleaseSnapshot(skills=skills, experts=target.experts)


def _validate_rollback_scope(
    state: DatabaseState, adoption: Adoption, snapshot: ReleaseSnapshot
) -> None:
    validate_scope(
        state,
        adoption,
        {slug for slug, skill in snapshot.skills.items() if not skill.retired},
        {slug for slug, skill in snapshot.skills.items() if skill.retired},
        set(snapshot.experts),
    )


def _snapshot_preview(
    release_id: str,
    revision: str,
    snapshot: ReleaseSnapshot,
    adoption: Adoption,
    state: DatabaseState,
) -> Preview:
    return Preview(
        database_target=state.database_target,
        release_id=release_id,
        rollback_release_id=state.active_release_id or release_id,
        revision=revision,
        previous_release_id=state.active_release_id,
        generation=state.generation,
        state_sha256=state.fingerprint(),
        adoption_sha256=digest(adoption.model_dump()),
        create_skills=[],
        update_skills=sorted(
            slug for slug, skill in snapshot.skills.items() if not skill.retired
        ),
        retire_skills=sorted(
            slug for slug, skill in snapshot.skills.items() if skill.retired
        ),
        expert_keys=sorted(snapshot.experts),
        activate_experts=sorted(
            key
            for key, expert in snapshot.experts.items()
            if state.experts[key].is_archived and not expert.is_archived
        ),
    )


def _validate_rollback_ids(snapshot: ReleaseSnapshot, adoption: Adoption) -> None:
    if any(
        adoption.skills.get(slug) != skill.listing_id
        for slug, skill in snapshot.skills.items()
    ):
        raise ValueError("rollback skill IDs disagree with the adopted records")
    if any(
        adoption.experts.get(key) != expert.expert_id
        for key, expert in snapshot.experts.items()
    ):
        raise ValueError("rollback expert IDs disagree with the adopted records")
