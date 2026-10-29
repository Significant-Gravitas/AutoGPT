"""Learning at the owner's saved-skill limit."""

from unittest.mock import AsyncMock

import pytest

from backend.copilot.learning import nightly, nightly_test, publish
from backend.copilot.learning.nightly_test import EXPERT, USER, _run, _source
from backend.copilot.tools.skills import parse_skill_markdown

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries

NAME = "csv-import-checks"


@pytest.fixture
def at_limit(monkeypatch):
    limit = AsyncMock(return_value=True)
    monkeypatch.setattr(publish, "saved_skill_limit_reached", limit)
    return limit


@pytest.mark.asyncio
async def test_a_new_skill_waits_for_room_and_is_published_from_its_kept_review(
    fake_store, adapter, boundaries, at_limit
):
    """A new skill at the limit leaves no head or pending write that could
    publish unprompted later; its paid review is kept and reused."""
    source = await _source(fake_store)

    waiting = await _run()

    assert waiting.dispositions == {"paused": 1}
    at_limit.assert_awaited_with(USER, EXPERT, NAME)
    assert await fake_store.get_head(USER, EXPERT, NAME) is None
    assert await fake_store.list_pending_publications(USER) == []
    boundaries["write"].assert_not_awaited()
    ledger = await fake_store.get_review_for_revision(
        USER, source.id, source.revision, nightly.POLICY_VERSION
    )
    assert ledger.replay is not None
    assert ledger.reason == publish.SKILL_LIMIT_REASON
    assert (await fake_store.get_source(USER, source.id)).processed_revision is None

    at_limit.return_value = False
    published = await _run()

    assert published.applied == 1 and published.model_calls == 0
    boundaries["review"].assert_awaited_once()
    head = await fake_store.get_head(USER, EXPERT, NAME)
    assert head is not None and head.current_version_id is not None


async def _publish_an_update(fake_store, monkeypatch):
    """Publish a skill, then propose an update to it through the publisher."""
    await _source(fake_store)
    assert (await _run()).applied == 1
    head = await fake_store.get_head(USER, EXPERT, NAME)
    current = await fake_store.get_version(USER, head.current_version_id)
    saved = parse_skill_markdown(current.content)
    monkeypatch.setattr(
        publish, "read_user_skill_markdown", AsyncMock(return_value=current.content)
    )
    request = publish.PublishRequest(
        user_id=USER,
        expert_id=EXPERT,
        skill_name=NAME,
        description=saved.description,
        triggers=list(saved.triggers),
        body=saved.body + "\n4. Log the rows that were skipped.\n",
        summary="Logged the rows that were skipped.",
        origin="saved_overnight",
    )
    return head, request


@pytest.mark.asyncio
async def test_an_update_that_needs_no_new_slot_applies_at_the_limit(
    fake_store, adapter, boundaries, at_limit, monkeypatch
):
    """Every learned publication asks; a skill that already fills its slot
    gets room, so it keeps improving."""
    at_limit.return_value = False
    _, request = await _publish_an_update(fake_store, monkeypatch)
    at_limit.reset_mock()

    outcome = await publish.publish_learned_version(request)

    assert outcome.status == "applied"
    at_limit.assert_awaited_once_with(USER, EXPERT, NAME)
    assert (await fake_store.get_head(USER, EXPERT, NAME)).current_version == 2


@pytest.mark.asyncio
async def test_an_update_that_would_take_a_new_slot_waits_like_a_new_skill(
    fake_store, adapter, boundaries, at_limit, monkeypatch
):
    """An existing root can still need a saved slot, for example an
    installed copy with no install record that an edit moves into the
    owner's budget. At the limit it waits too, before any version."""
    at_limit.return_value = False
    head, request = await _publish_an_update(fake_store, monkeypatch)
    writes = boundaries["write"].await_count
    at_limit.return_value = True

    outcome = await publish.publish_learned_version(request)

    assert (outcome.status, outcome.reason) == ("paused", publish.SKILL_LIMIT_REASON)
    assert await fake_store.get_head(USER, EXPERT, NAME) == head
    assert len(await fake_store.list_versions(USER, EXPERT, NAME)) == 1
    assert await fake_store.list_pending_publications(USER) == []
    assert boundaries["write"].await_count == writes
