from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from . import nightly, nightly_test
from .publication_models import PublishOutcome

adapter = nightly_test.adapter
boundaries = nightly_test.boundaries


async def _backlog(store):
    sources = [
        await nightly_test._source(store, ref=f"session-{index}")
        for index in range(nightly.MAX_REVIEWS_PER_OWNER + 1)
    ]
    for index, source in enumerate(sources):
        store.sources[source.id] = source.model_copy(
            update={
                "updated_at": datetime(2020, 1, 1, tzinfo=timezone.utc)
                + timedelta(seconds=index)
            }
        )
    return sources


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["provider_error", "adapter_exception", "pending_write", "conflict", "deferred"],
)
async def test_retryable_sources_yield_priority_without_advancing_cursors(
    fake_store, adapter, boundaries, monkeypatch, failure
):
    sources = await _backlog(fake_store)
    if failure == "provider_error":
        boundaries["review"].side_effect = nightly_test.DreamLLMError("outage")
    elif failure == "adapter_exception":

        async def fail(source):
            raise RuntimeError("source adapter unavailable")

        adapter.on_load.append(fail)
    elif failure == "pending_write":
        boundaries["write"].side_effect = OSError("storage unavailable")
    elif failure == "conflict":
        monkeypatch.setattr(
            nightly,
            "publish_learned_version",
            AsyncMock(return_value=PublishOutcome(status="conflict")),
        )
    elif failure == "deferred":
        monkeypatch.setattr(
            nightly,
            "_open_proposals",
            AsyncMock(return_value=nightly.MAX_OPEN_PROPOSALS_PER_OWNER),
        )

    result = await nightly_test._run()
    assert result.applied == 0
    pending = await nightly._select_sources(nightly_test.USER, None)
    assert len(pending) == nightly.MAX_REVIEWS_PER_OWNER
    assert pending[0].id == sources[-1].id
    for source in sources:
        current = await fake_store.get_source(nightly_test.USER, source.id)
        assert current.processed_revision is None
        assert current.revision == source.revision
        assert current.epoch == source.epoch


@pytest.mark.asyncio
async def test_budget_skipped_sources_keep_priority(fake_store, adapter, boundaries):
    sources = await _backlog(fake_store)
    before = dict(fake_store.sources)
    boundaries["budget"].side_effect = [(True, None), (False, "daily_budget_reached")]
    result = await nightly_test._run()
    assert result.model_calls == 0
    pending = await nightly._select_sources(nightly_test.USER, None)
    assert pending[0].id == sources[1].id
    for source in sources[1:]:
        assert fake_store.sources[source.id].updated_at == before[source.id].updated_at
