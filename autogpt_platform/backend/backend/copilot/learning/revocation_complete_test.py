from unittest.mock import AsyncMock

import pytest

from . import publish_test, revocation
from .publish_test import EXPERT, KIND, NAME, USER, _seed

workspace = publish_test.workspace


async def _seed_dependent(store):
    source = await store.upsert_source_revision(
        USER,
        expert_id=EXPERT,
        source_kind=KIND,
        source_id="withdrawn",
        revision="1",
        evidence_refs=[],
        outcome_signals=[],
    )
    current, independent = await _seed(store)
    store.versions[current.id] = current.model_copy(
        update={
            "sources": [{"source_id": source.id}],
            "base_version_id": independent.id,
        }
    )
    return source, current, independent


@pytest.mark.asyncio
async def test_revocation_finds_old_current_and_fallback_beyond_history_cap(
    fake_store, workspace
):
    source, current, independent = await _seed_dependent(fake_store)
    noisy = await fake_store.ensure_head(USER, None, "frequently-edited")
    for n in range(601):
        await fake_store.create_version(
            USER,
            head=noisy,
            content=f"Edit {n}",
            description="Noise",
            triggers=[],
            origin="edited",
        )
    affected = await revocation.invalidate_versions_for_source(
        USER, source, "withdrawn"
    )
    assert current.id in affected
    head = await fake_store.get_head(USER, EXPERT, NAME)
    restored = await fake_store.get_version(USER, head.current_version_id)
    assert restored.restored_from_version_id == independent.id
    assert restored.state == "ready" and head.use_paused_at is None
    assert (
        len(
            await fake_store.list_versions(
                USER, "personal", noisy.skill_name, limit=1000
            )
        )
        == 601
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_paused", [False, True])
async def test_retry_after_repair_exception_recovers_without_overriding_owner_pause(
    fake_store, workspace, monkeypatch, owner_paused
):
    source, current, independent = await _seed_dependent(fake_store)
    await fake_store.update_head_policy(USER, EXPERT, NAME, use_paused=owner_paused)
    original = revocation.restore_version
    broken = AsyncMock(side_effect=RuntimeError("temporary database outage"))
    monkeypatch.setattr(revocation, "restore_version", broken)
    with pytest.raises(RuntimeError, match="temporary database outage"):
        await revocation.invalidate_versions_for_source(USER, source, "withdrawn")
    assert (await fake_store.get_version(USER, current.id)).state == "invalidated"
    monkeypatch.setattr(revocation, "restore_version", original)
    assert current.id in await revocation.invalidate_versions_for_source(
        USER, source, "withdrawn"
    )
    head = await fake_store.get_head(USER, EXPERT, NAME)
    restored = await fake_store.get_version(USER, head.current_version_id)
    assert restored.restored_from_version_id == independent.id
    assert (head.use_paused_at is not None) == owner_paused
    await revocation.invalidate_versions_for_source(USER, source, "withdrawn")
    after = await fake_store.get_head(USER, EXPERT, NAME)
    assert after.current_version_id == restored.id
    assert (await fake_store.get_version(USER, restored.id)).state == "ready"
