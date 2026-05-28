"""Tests for the BatchExecutor poll loop + Redis pending queue."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.executor.batch_executor import (
    INITIAL_POLL_DELAY_SECONDS,
    MAX_BATCH_LIFETIME_SECONDS,
    MAX_POLL_DELAY_SECONDS,
    PendingEntry,
    _bump_delay,
    _next_poll_at,
    clear_handlers_for_test,
    enqueue_pending,
    list_pending,
    register_handler,
    remove_pending,
    update_pending,
    walk_once,
)
from backend.util.llm.providers import BatchResultRow


@pytest.fixture
def fake_redis():
    """In-memory hash-backed redis stub."""
    store: dict[str, dict[str, str]] = {}

    async def fake_hset(key, field, value):
        store.setdefault(key, {})[field] = value
        return 1

    async def fake_hgetall(key):
        return store.get(key, {})

    async def fake_hdel(key, field):
        if key in store and field in store[key]:
            del store[key][field]
            return 1
        return 0

    async def fake_delete(key):
        store.pop(key, None)

    async def fake_expire(key, ttl):
        return 1

    stub = AsyncMock()
    stub.hset.side_effect = fake_hset
    stub.hgetall.side_effect = fake_hgetall
    stub.hdel.side_effect = fake_hdel
    stub.delete.side_effect = fake_delete
    stub.expire.side_effect = fake_expire

    async def fake_get_redis_async():
        return stub

    with patch(
        "backend.data.redis_client.get_redis_async",
        side_effect=fake_get_redis_async,
    ):
        yield stub, store


@pytest.fixture(autouse=True)
def reset_handlers():
    """Reset the in-memory handler registry between tests."""
    clear_handlers_for_test()
    yield
    clear_handlers_for_test()


def _entry(
    *,
    provider_batch_id: str = "msgbatch_1",
    namespace: str = "dream_pass",
    next_poll_at: datetime | None = None,
    submitted_at: datetime | None = None,
    delay: int = INITIAL_POLL_DELAY_SECONDS,
    payload: dict[str, Any] | None = None,
) -> PendingEntry:
    now = datetime.now(timezone.utc)
    return PendingEntry(
        provider="anthropic",
        provider_batch_id=provider_batch_id,
        callback_namespace=namespace,
        submitted_at=submitted_at or now,
        next_poll_at=next_poll_at or now,
        poll_delay_seconds=delay,
        payload=payload or {"custom_ids": ["passid-1:consolidate"]},
    )


class TestQueueRoundtrip:
    @pytest.mark.asyncio
    async def test_enqueue_then_list_returns_entry(self, fake_redis):
        await enqueue_pending(_entry())
        entries = await list_pending()
        assert len(entries) == 1
        assert entries[0].provider_batch_id == "msgbatch_1"
        assert entries[0].callback_namespace == "dream_pass"

    @pytest.mark.asyncio
    async def test_remove_drops_entry(self, fake_redis):
        await enqueue_pending(_entry())
        await remove_pending("msgbatch_1")
        assert await list_pending() == []

    @pytest.mark.asyncio
    async def test_update_pending_overwrites(self, fake_redis):
        entry = _entry(delay=30)
        await enqueue_pending(entry)
        entry.poll_delay_seconds = 120
        entry.next_poll_at = datetime.now(timezone.utc) + timedelta(seconds=120)
        await update_pending(entry)
        result = (await list_pending())[0]
        assert result.poll_delay_seconds == 120

    @pytest.mark.asyncio
    async def test_payload_roundtrips_through_json(self, fake_redis):
        """Payload is opaque to the executor but must survive
        serialization — phase mappings and pass_id ride in here."""
        e = _entry(
            payload={
                "user_id": "u",
                "pass_id": "p",
                "job_id": "j",
                "phase_for_custom_id": {"p:consolidate": "consolidate"},
            }
        )
        await enqueue_pending(e)
        back = (await list_pending())[0]
        assert back.payload["user_id"] == "u"
        assert back.payload["phase_for_custom_id"] == {"p:consolidate": "consolidate"}


class TestWalkOnceDispatch:
    @pytest.mark.asyncio
    async def test_ended_batch_downloads_dispatches_removes(self, fake_redis):
        seen_rows: list[list[BatchResultRow]] = []

        async def fake_handler(entry, rows):
            seen_rows.append(list(rows))

        register_handler("dream_pass", fake_handler)
        await enqueue_pending(_entry())

        fake_row = BatchResultRow(
            custom_id="passid-1:consolidate",
            content='{"facts": []}',
            input_tokens=10,
            output_tokens=20,
        )
        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="ended"),
        ), patch(
            "backend.executor.batch_executor.download_batch_results",
            new=AsyncMock(return_value=[fake_row]),
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        assert len(seen_rows) == 1
        assert seen_rows[0][0].custom_id == "passid-1:consolidate"
        # Entry must be removed after successful dispatch.
        assert await list_pending() == []

    @pytest.mark.asyncio
    async def test_processing_batch_bumps_delay_and_keeps_entry(self, fake_redis):
        register_handler("dream_pass", AsyncMock())
        await enqueue_pending(_entry(delay=30))
        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="processing"),
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        entries = await list_pending()
        assert len(entries) == 1
        # Backoff doubled the delay
        assert entries[0].poll_delay_seconds == 60

    @pytest.mark.asyncio
    async def test_failed_batch_dispatches_synthetic_error_and_removes(
        self, fake_redis
    ):
        captured: list[list[BatchResultRow]] = []

        async def fake_handler(entry, rows):
            captured.append(list(rows))

        register_handler("dream_pass", fake_handler)
        await enqueue_pending(_entry())

        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="failed"),
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        assert len(captured) == 1
        assert captured[0][0].error == "provider reported failed"
        assert await list_pending() == []

    @pytest.mark.asyncio
    async def test_skips_entry_whose_next_poll_is_in_future(self, fake_redis):
        register_handler("dream_pass", AsyncMock())
        future_time = datetime.now(timezone.utc) + timedelta(seconds=300)
        await enqueue_pending(_entry(next_poll_at=future_time))
        poll_mock = AsyncMock(return_value="ended")
        with patch("backend.executor.batch_executor.poll_batch", poll_mock):
            await walk_once(api_key_for=lambda p: "sk-ant-test")
        poll_mock.assert_not_awaited()
        # Entry still pending, no state change
        assert len(await list_pending()) == 1

    @pytest.mark.asyncio
    async def test_expired_entry_dispatches_timeout_and_removes(self, fake_redis):
        """Entries older than ``MAX_BATCH_LIFETIME_SECONDS`` get a
        synthetic timeout error so the JobStatus row doesn't sit at
        ``submitted`` forever even when the provider goes silent."""
        captured: list[list[BatchResultRow]] = []

        async def fake_handler(entry, rows):
            captured.append(list(rows))

        register_handler("dream_pass", fake_handler)
        long_ago = datetime.now(timezone.utc) - timedelta(
            seconds=MAX_BATCH_LIFETIME_SECONDS + 1
        )
        await enqueue_pending(_entry(submitted_at=long_ago))

        poll_mock = AsyncMock()
        with patch("backend.executor.batch_executor.poll_batch", poll_mock):
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        poll_mock.assert_not_awaited()
        assert captured[0][0].error == (
            "exceeded MAX_BATCH_LIFETIME_SECONDS without completion"
        )
        assert await list_pending() == []

    @pytest.mark.asyncio
    async def test_no_api_key_leaves_entry_in_queue(self, fake_redis):
        """When the api_key_for factory returns None (rotation in
        progress, key removed), we MUST leave the entry alone so the
        retry catches it after the key is restored."""
        register_handler("dream_pass", AsyncMock())
        await enqueue_pending(_entry())
        poll_mock = AsyncMock()
        with patch("backend.executor.batch_executor.poll_batch", poll_mock):
            await walk_once(api_key_for=lambda p: None)
        poll_mock.assert_not_awaited()
        # Entry must still be there for the next walk to retry.
        assert len(await list_pending()) == 1

    @pytest.mark.asyncio
    async def test_poll_exception_bumps_delay_and_keeps_entry(self, fake_redis):
        """A transient provider-side failure on poll mustn't lose the
        entry — back off and try again."""
        register_handler("dream_pass", AsyncMock())
        await enqueue_pending(_entry(delay=30))
        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(side_effect=RuntimeError("upstream 502")),
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")
        entries = await list_pending()
        assert len(entries) == 1
        assert entries[0].poll_delay_seconds == 60


class TestUnknownNamespace:
    @pytest.mark.asyncio
    async def test_no_handler_drops_results_logs_warning(self, fake_redis):
        """An ended batch whose namespace nobody registered for must
        NOT crash the walk — it logs + drops so the rest of the queue
        still progresses."""
        await enqueue_pending(_entry(namespace="orphan_namespace"))
        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="ended"),
        ), patch(
            "backend.executor.batch_executor.download_batch_results",
            new=AsyncMock(
                return_value=[
                    BatchResultRow(
                        custom_id="x",
                        content="x",
                        input_tokens=1,
                        output_tokens=1,
                    )
                ]
            ),
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")
        # The entry was processed (downloaded) and removed even with
        # no handler — caller couldn't have known the namespace would
        # be unbound at result time.
        assert await list_pending() == []


class TestBackoffMath:
    def test_bump_delay_doubles_until_cap(self):
        assert _bump_delay(30) == 60
        assert _bump_delay(60) == 120
        assert _bump_delay(120) == 240
        assert _bump_delay(150) == 300
        assert _bump_delay(MAX_POLL_DELAY_SECONDS) == MAX_POLL_DELAY_SECONDS
        # Even way past the cap, stays capped
        assert _bump_delay(MAX_POLL_DELAY_SECONDS * 10) == MAX_POLL_DELAY_SECONDS

    def test_next_poll_at_is_in_future(self):
        before = datetime.now(timezone.utc)
        result = _next_poll_at(60)
        after = datetime.now(timezone.utc) + timedelta(seconds=60)
        assert before + timedelta(seconds=59) <= result <= after + timedelta(seconds=1)
