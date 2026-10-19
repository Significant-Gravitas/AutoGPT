from uuid import uuid4

import pytest

from backend.data import skill_learning as learning
from backend.data.skill_learning_test import _cleanup, _create_user, _source
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_attempted_sources_yield_to_pending_work(server: SpinTestServer):
    user = str(uuid4())
    await _create_user(user)
    try:
        sources = [await _source(user) for _ in range(9)]
        first = await learning.list_pending_sources(user, owner_key="personal", limit=8)
        assert {s.id for s in first} == {s.id for s in sources[:8]}
        for source in first:
            assert await learning.mark_source_attempted(
                user, source.id, source.revision
            )
        next_page = await learning.list_pending_sources(
            user, owner_key="personal", limit=8
        )
        assert next_page[0].id == sources[-1].id
        for source in sources:
            current = await learning.require_owned_source(user, source.id)
            assert current.processed_revision is None
            assert current.revision == source.revision
            assert current.epoch == source.epoch
    finally:
        await _cleanup(user)


@pytest.mark.asyncio(loop_scope="session")
async def test_attempt_priority_update_checks_user_and_revision(server: SpinTestServer):
    user = str(uuid4())
    await _create_user(user)
    try:
        source = await _source(user)
        assert not await learning.mark_source_attempted(str(uuid4()), source.id, "1")
        assert await learning.require_owned_source(user, source.id) == source
        newer = await learning.upsert_source_revision(
            user,
            expert_id=None,
            source_kind=source.source_kind,
            source_id=source.source_id,
            revision="2",
            evidence_refs=[],
            outcome_signals=[],
        )
        assert not await learning.mark_source_attempted(user, source.id, "1")
        assert await learning.require_owned_source(user, source.id) == newer
        assert await learning.mark_source_attempted(user, source.id, "2")
        current = await learning.require_owned_source(user, source.id)
        assert current.processed_revision is None
        assert current.revision == newer.revision
        assert current.epoch == newer.epoch
    finally:
        await _cleanup(user)
