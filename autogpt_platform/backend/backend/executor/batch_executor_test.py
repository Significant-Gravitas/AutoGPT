"""Tests for the BatchExecutor poll loop + Redis pending queue."""

from __future__ import annotations

import json
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
    _dispatch_error,
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
    """In-memory hash-backed redis stub.

    Also fakes the Lua ``eval`` path so the BatchExecutor's atomic
    ``claim_batch_dispatch_atomic`` works under test — the real script
    SETs a per-batch tombstone key (``NX EX``) + HDELs the pending
    entry in one indivisible step; the fake replays the same effects.
    """
    store: dict[str, dict[str, str]] = {}
    tombstones: dict[str, str] = {}

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

    async def fake_eval(script, numkeys, *args):
        # Only the claim_batch_dispatch_atomic shape is used by the
        # BatchExecutor today. Branch on content so we don't pretend
        # to know how to evaluate arbitrary Lua.
        if "HDEL" in script:
            pending_key, tombstone_key = args[0], args[1]
            batch_id = args[2]
            if tombstone_key in tombstones:
                return 0
            tombstones[tombstone_key] = "1"
            store.setdefault(pending_key, {}).pop(batch_id, None)
            return 1
        raise NotImplementedError(f"fake_redis.eval: unknown script: {script!r}")

    stub = AsyncMock()
    stub.hset.side_effect = fake_hset
    stub.hgetall.side_effect = fake_hgetall
    stub.hdel.side_effect = fake_hdel
    stub.delete.side_effect = fake_delete
    stub.expire.side_effect = fake_expire
    stub.eval.side_effect = fake_eval

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
    async def test_redispatch_after_claim_is_skipped_and_zombie_cleared(
        self, fake_redis
    ):
        """Second walk that races on the same provider_batch_id must
        not re-invoke the namespace handler — the atomic claim is the
        only thing standing between "BatchExecutor saw the same ended
        batch twice" and "user's billing/apply/phase-chain side effects
        re-fire". Regression for the double-sanitize incident in prod.
        The refused-claim zombie row must also be cleared — left alone
        it would be re-polled AND re-downloaded every walk until the
        lifetime timeout reaps it (up to ~24h).
        """
        seen_rows: list[list[BatchResultRow]] = []

        async def fake_handler(entry, rows):
            seen_rows.append(list(rows))

        register_handler("dream_pass", fake_handler)
        await enqueue_pending(_entry())

        fake_row = BatchResultRow(
            custom_id="passid-1:consolidate",
            content='{"facts": []}',
            input_tokens=1,
            output_tokens=1,
        )
        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="ended"),
        ), patch(
            "backend.executor.batch_executor.download_batch_results",
            new=AsyncMock(return_value=[fake_row]),
        ):
            # First walk: claim wins, handler fires.
            await walk_once(api_key_for=lambda p: "sk-ant-test")
            # Simulate a second walk that races (or a crash-recovery
            # replay) — the same entry is re-added to pending, but the
            # claim must refuse so the handler doesn't fire again.
            await enqueue_pending(_entry())
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        assert (
            len(seen_rows) == 1
        ), "handler should fire exactly once even on re-dispatch"
        # The refused-claim zombie row is removed without dispatching
        # so subsequent walks don't keep re-polling + re-downloading.
        assert await list_pending() == []

    @pytest.mark.asyncio
    async def test_failed_batch_refused_claim_clears_zombie_without_dispatch(
        self, fake_redis
    ):
        """Same zombie cleanup on the failed path: a refused claim
        proves the batch already dispatched once, so the resurrected
        pending row is cleared instead of being re-polled every walk
        until the lifetime timeout."""
        captured: list[list[BatchResultRow]] = []

        async def fake_handler(entry, rows):
            captured.append(list(rows))

        register_handler("dream_pass", fake_handler)
        await enqueue_pending(_entry())

        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="failed"),
        ):
            # First walk: claim wins, synthetic error dispatches once.
            await walk_once(api_key_for=lambda p: "sk-ant-test")
            # Crash-replay resurrects the entry; the claim must refuse
            # and the zombie row must be cleared without dispatching.
            await enqueue_pending(_entry())
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        assert len(captured) == 1, "error handler must fire exactly once"
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
    async def test_timeout_dispatch_fires_exactly_once_across_replays(self, fake_redis):
        """The timeout path must claim atomically before dispatching —
        a crash-replay (entry re-added after the first dispatch) or a
        racing second walker must not re-fire the handler. A double
        fire would release the user's dream lock twice, potentially
        deleting a NEWER pass's lock acquired after the first release.
        """
        captured: list[list[BatchResultRow]] = []

        async def fake_handler(entry, rows):
            captured.append(list(rows))

        register_handler("dream_pass", fake_handler)
        long_ago = datetime.now(timezone.utc) - timedelta(
            seconds=MAX_BATCH_LIFETIME_SECONDS + 1
        )
        await enqueue_pending(_entry(submitted_at=long_ago))
        await walk_once(api_key_for=lambda p: "sk-ant-test")
        # Crash-replay: the same entry reappears in pending after the
        # first timeout dispatch already claimed the batch.
        await enqueue_pending(_entry(submitted_at=long_ago))
        await walk_once(api_key_for=lambda p: "sk-ant-test")

        assert len(captured) == 1, "timeout handler must fire exactly once"
        # The replayed zombie row is cleared without dispatching so it
        # can't be re-walked until the tombstone expires.
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
    async def test_no_handler_leaves_entry_pending_for_recovery(self, fake_redis):
        """A batch whose namespace nobody registered for must NOT be
        claimed + dropped — the tombstone would make the results
        irrecoverable even though the provider retains them for weeks.
        Leave the entry pending (with backoff) so a deploy that fixes
        the handler registration can still dispatch it."""
        await enqueue_pending(_entry(namespace="orphan_namespace", delay=30))
        poll_mock = AsyncMock(return_value="ended")
        download_mock = AsyncMock(return_value=[])
        with patch(
            "backend.executor.batch_executor.poll_batch",
            poll_mock,
        ), patch(
            "backend.executor.batch_executor.download_batch_results",
            download_mock,
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        # We don't even poll: there's nobody to deliver results to.
        poll_mock.assert_not_awaited()
        download_mock.assert_not_awaited()
        entries = await list_pending()
        assert len(entries) == 1
        # Backed off, not hot-looping the error every walk.
        assert entries[0].poll_delay_seconds == 60

    @pytest.mark.asyncio
    async def test_no_handler_entry_is_reaped_by_lifetime_timeout(self, fake_redis):
        """Leaving handler-less entries pending can't loop forever:
        once the entry crosses MAX_BATCH_LIFETIME_SECONDS the timeout
        path claims + removes it even with no handler registered
        (_dispatch_error no-ops when the handler is absent)."""
        long_ago = datetime.now(timezone.utc) - timedelta(
            seconds=MAX_BATCH_LIFETIME_SECONDS + 1
        )
        await enqueue_pending(
            _entry(namespace="orphan_namespace", submitted_at=long_ago)
        )
        await walk_once(api_key_for=lambda p: "sk-ant-test")
        assert await list_pending() == []


class TestWalkCrashGuard:
    @pytest.mark.asyncio
    async def test_crash_on_one_entry_does_not_starve_later_entries(self, fake_redis):
        """An uncaught per-entry error (here: the api-key factory
        blowing up) must be contained to that entry — entries after it
        in hash order still get polled and dispatched, and the bad
        entry stays pending for the next walk instead of aborting the
        whole queue forever."""
        dispatched: list[str] = []

        async def fake_handler(entry, rows):
            dispatched.append(entry.provider_batch_id)

        register_handler("dream_pass", fake_handler)
        await enqueue_pending(_entry(provider_batch_id="msgbatch_poison"))
        await enqueue_pending(_entry(provider_batch_id="msgbatch_healthy"))

        calls = {"count": 0}

        def flaky_api_key_for(provider):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("settings backend exploded")
            return "sk-ant-test"

        fake_row = BatchResultRow(
            custom_id="passid-1:consolidate",
            content='{"facts": []}',
            input_tokens=1,
            output_tokens=1,
        )
        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="ended"),
        ), patch(
            "backend.executor.batch_executor.download_batch_results",
            new=AsyncMock(return_value=[fake_row]),
        ):
            await walk_once(api_key_for=flaky_api_key_for)

        assert dispatched == ["msgbatch_healthy"]
        # The poisoned entry is untouched, ready for the next walk.
        remaining = [e.provider_batch_id for e in await list_pending()]
        assert remaining == ["msgbatch_poison"]

    @pytest.mark.asyncio
    async def test_naive_submitted_at_in_redis_is_coerced_and_walked(self, fake_redis):
        """A producer that wrote naive isoformat timestamps must not
        poison the walk — PendingEntry coerces naive datetimes to UTC
        on rehydration, so ``now - submitted_at`` can't raise TypeError
        and brick every entry behind it in hash order."""
        stub, store = fake_redis
        naive_iso = (
            (datetime.now(timezone.utc) - timedelta(seconds=60))
            .replace(tzinfo=None)
            .isoformat()
        )
        store["{llm:batch}:pending"] = {
            "msgbatch_naive": json.dumps(
                {
                    "provider": "anthropic",
                    "provider_batch_id": "msgbatch_naive",
                    "callback_namespace": "dream_pass",
                    "submitted_at": naive_iso,
                    "next_poll_at": naive_iso,
                    "poll_delay_seconds": 30,
                    "payload": {"custom_ids": ["x"]},
                }
            )
        }
        register_handler("dream_pass", AsyncMock())

        entries = await list_pending()
        assert entries[0].submitted_at.tzinfo == timezone.utc
        assert entries[0].next_poll_at.tzinfo == timezone.utc

        with patch(
            "backend.executor.batch_executor.poll_batch",
            new=AsyncMock(return_value="processing"),
        ):
            await walk_once(api_key_for=lambda p: "sk-ant-test")

        # Walked normally: still pending with the delay backed off.
        assert (await list_pending())[0].poll_delay_seconds == 60

    def test_naive_datetimes_are_coerced_to_utc_on_construction(self):
        naive = datetime(2026, 1, 1, 12, 0, 0)
        entry = PendingEntry(
            provider="anthropic",
            provider_batch_id="msgbatch_x",
            callback_namespace="dream_pass",
            submitted_at=naive,
            next_poll_at=naive,
        )
        assert entry.submitted_at.tzinfo == timezone.utc
        assert entry.next_poll_at.tzinfo == timezone.utc


class TestDispatchErrorGuard:
    @pytest.mark.asyncio
    async def test_malformed_custom_ids_payload_is_contained(self):
        """``payload`` is an open contract — junk ``custom_ids`` (here
        an int instead of a list) must be caught inside _dispatch_error
        rather than escaping after the batch was already claimed."""
        handler = AsyncMock()
        register_handler("dream_pass", handler)
        entry = _entry(payload={"custom_ids": 123})

        await _dispatch_error(entry, error="provider reported failed")

        handler.assert_not_awaited()


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


def test_standalone_entry_point_runs_only_the_batch_executor():
    """k8s deploys services individually; the infra repo needs a
    console script that runs the BatchExecutor on its own (mirrors
    backend/scheduler.py's shape)."""
    import backend.batch_executor as entry
    from backend.executor.batch_executor import BatchExecutor

    assert callable(entry.main)
    assert entry.BatchExecutor is BatchExecutor

    with patch.object(entry, "run_processes") as run_processes:
        entry.main()

    run_processes.assert_called_once()
    (batch_executor,) = run_processes.call_args.args
    assert isinstance(batch_executor, BatchExecutor)
