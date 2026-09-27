"""The reaper as the backstop for every stopped pass, over the in-memory
store, Redis and the executor's queue. A cancel or a guard's expiry marks the
row and keeps its lease token in the transition that closes it, so when
nothing else cleans up after the pass (the executor down and never polling;
a drop hook that raises, times out or dies after its claim) the reaper does,
once the pass has had its grace to stop: the batch stopped at the provider,
the landed phase charged once, the lock released under the kept token, the
state and bundle deleted, the mark cleared. Before that grace it leaves the
lock of a pass that may still be running alone; a pass that cleans up after
itself clears its own mark. The provider and the charges are stubbed at
their edges."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from prisma.enums import DreamPassStatus

from backend.executor import batch_executor as executor

from . import batch_deliveries
from . import reaper as reaper_mod
from .batch_callbacks import handle_dream_batch_result
from .batch_deliveries_test import (
    _LOCK_KEY,
    _MODELS,
    _emulate_the_dispatch_claim,
    _entry,
    _in_flight,
    _row,
)
from .batch_state import state_key
from .batch_submit import input_bundle_key
from .cancel import cancel_dream_pass
from .conftest import FakeAsyncRedis, FakeDreamDb
from .locks import BATCH_LOCK_TTL_SECONDS
from .pass_record import expired
from .reaper import REAP_GRACE_SECONDS, reap_expired_passes
from .reaper_test import _charged
from .schemas import DreamOperations
from .store import write_stop


@pytest.fixture(autouse=True)
def provider(mocker) -> AsyncMock:
    mocker.patch(
        "backend.copilot.dream.provider_batch.anthropic_api_key", return_value="k"
    )
    mocker.patch.object(reaper_mod, "phase_models_for_config", return_value=_MODELS)
    return mocker.patch(
        "backend.copilot.dream.provider_batch.cancel_batch",
        AsyncMock(return_value=True),
    )


@pytest.fixture(autouse=True)
def charges(mocker) -> AsyncMock:
    return mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )


@pytest.fixture(autouse=True)
def handlers():
    executor.clear_handlers_for_test()
    yield
    executor.clear_handlers_for_test()


class TestACancelledPassNobodyPolls:
    async def test_is_left_alone_for_its_grace_then_cleaned_up_by_the_reaper(
        self, fake_dream_db, fake_dream_redis, provider, charges
    ):
        """Codex's case: a SUBMITTED pass cancelled while the executor is
        down. Its lease is a batch lease, fresh for a day; the reaper waits
        out the grace from the cancel, time for a pass still running to see
        it, then does what the drop would have."""
        await _submitted(fake_dream_db, fake_dream_redis)
        await executor.enqueue_pending(_entry("recombine"))

        assert (await cancel_dream_pass("p1", user_id="u1", reason="stop")).cancelled

        _assert_marked(fake_dream_db, DreamPassStatus.CANCELLED)
        provider.assert_awaited_once()
        within_the_grace = await reap_expired_passes(
            now=_after(REAP_GRACE_SECONDS - 60)
        )
        assert within_the_grace.listed == 0
        assert fake_dream_redis.store[_LOCK_KEY] == "tok"

        run = await reap_expired_passes(now=_after(REAP_GRACE_SECONDS + 60))

        assert run.outcomes == {"cleaned": 1}
        assert provider.await_count == 2
        assert provider.await_args.kwargs["provider_batch_id"] == "b-rec"
        _assert_cleaned_up(fake_dream_db, fake_dream_redis, charges)


class TestADropHookThatDoesNotFinish:
    @pytest.mark.parametrize("failure", ["exception", "timeout", "crash_after_claim"])
    async def test_leaves_the_pass_to_the_reaper(
        self, monkeypatch, fake_dream_db, fake_dream_redis, charges, failure
    ):
        """Codex's three: the executor claims the entry off its queue, and
        its hook raises, runs out of time, or the process dies right after
        the claim. The row keeps the mark the cancel set, and the reaper
        finishes the cleanup."""
        await _submitted(fake_dream_db, fake_dream_redis)
        _emulate_the_dispatch_claim(monkeypatch, fake_dream_redis)
        entry = _entry("recombine")
        await executor.enqueue_pending(entry)
        assert (await cancel_dream_pass("p1", user_id="u1", reason="stop")).cancelled

        async def drop_that_fails(_entry) -> None:
            if failure == "exception":
                raise RuntimeError("drop hook failed")
            await asyncio.Event().wait()

        executor.register_handler(
            "dream_pass",
            handle_dream_batch_result,
            should_dispatch=batch_deliveries.should_dispatch,
            on_drop=drop_that_fails,
        )
        monkeypatch.setattr(executor, "DISPATCH_CHECK_TIMEOUT_SECONDS", 0.05)
        if failure == "crash_after_claim":
            dies = AsyncMock(side_effect=asyncio.CancelledError("process died"))
            monkeypatch.setattr(executor, "_run_drop_hook", dies)
            with pytest.raises(asyncio.CancelledError):
                await _walk(entry)
        else:
            await _walk(entry)

        assert await executor.list_pending() == []
        charges.assert_not_awaited()
        _assert_marked(fake_dream_db, DreamPassStatus.CANCELLED)

        run = await reap_expired_passes(now=_after(REAP_GRACE_SECONDS + 60))

        assert run.outcomes == {"cleaned": 1}
        _assert_cleaned_up(fake_dream_db, fake_dream_redis, charges)


class TestAGuardExpiredBatchPass:
    async def test_is_cleaned_up_by_the_reaper_the_newer_passs_lock_kept(
        self, fake_dream_db, fake_dream_redis, provider, charges
    ):
        """A newer pass's guard expires a stale batch row while holding the
        scope's lock itself: the guard stops nothing at the provider, so the
        reaper does, charges the landed phase and cleans up; its unlock
        under the expired pass's token leaves the newer pass's lock alone."""
        await _submitted(fake_dream_db, fake_dream_redis)
        fake_dream_db.rows["p1"]["lease_expires_at"] = _after(-600)
        fake_dream_redis.store[_LOCK_KEY] = "newer-token"

        assert await write_stop("p1", expired("stale", not_updated_since=None))

        _assert_marked(fake_dream_db, DreamPassStatus.EXPIRED)
        provider.assert_not_awaited()

        run = await reap_expired_passes(now=_after(REAP_GRACE_SECONDS + 60))

        assert run.outcomes == {"cleaned": 1}
        provider.assert_awaited_once()
        assert _charged(charges) == ["consolidate"]
        assert fake_dream_redis.store[_LOCK_KEY] == "newer-token"
        assert state_key("p1") not in fake_dream_redis.hashes
        row = fake_dream_db.rows["p1"]
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)


class TestAPassThatEndsItself:
    async def test_clears_its_own_mark_once_its_cleanup_is_done(
        self, mocker, fake_dream_db, fake_dream_redis, charges
    ):
        """The applied tail closes the row marked, cleans up, and clears the
        mark itself: nothing is left for the reaper."""
        mocker.patch(
            "backend.copilot.dream.apply.apply_operations",
            AsyncMock(return_value={"writes": 0}),
        )
        await _in_flight(fake_dream_db, fake_dream_redis, landed=2)

        await handle_dream_batch_result(
            _entry("sanitize"), [_row("sanitize", DreamOperations())]
        )

        row = fake_dream_db.rows["p1"]
        assert row["status"] is DreamPassStatus.COMPLETE
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)
        assert _charged(charges) == ["consolidate", "recombine", "sanitize"]
        assert _LOCK_KEY not in fake_dream_redis.store
        assert (await reap_expired_passes(now=_after(86_400))).listed == 0

    async def test_whose_unlock_fails_leaves_its_mark_for_the_reaper(
        self, mocker, monkeypatch, fake_dream_db, fake_dream_redis, charges
    ):
        mocker.patch(
            "backend.copilot.dream.apply.apply_operations",
            AsyncMock(return_value={"writes": 0}),
        )
        await _in_flight(fake_dream_db, fake_dream_redis, landed=2)
        evaluate = fake_dream_redis.eval

        async def unlock_down(script: str, numkeys: int, *args):
            if '"del"' in script:
                raise ConnectionError("redis down at the unlock")
            return await evaluate(script, numkeys, *args)

        monkeypatch.setattr(fake_dream_redis, "eval", unlock_down)
        await handle_dream_batch_result(
            _entry("sanitize"), [_row("sanitize", DreamOperations())]
        )
        monkeypatch.setattr(fake_dream_redis, "eval", evaluate)

        _assert_marked(fake_dream_db, DreamPassStatus.COMPLETE)
        assert fake_dream_redis.store[_LOCK_KEY] == "tok"

        run = await reap_expired_passes(now=_after(REAP_GRACE_SECONDS + 60))

        assert run.outcomes == {"cleaned": 1}
        assert _LOCK_KEY not in fake_dream_redis.store
        assert _charged(charges) == ["consolidate", "recombine", "sanitize"]
        row = fake_dream_db.rows["p1"]
        assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)


async def _submitted(fake_dream_db: FakeDreamDb, fake_dream_redis: FakeAsyncRedis):
    """Pass p1 waiting on recombine's batch, consolidate landed, holding its
    lock and its lease for a batch's lifetime."""
    await _in_flight(fake_dream_db, fake_dream_redis)
    await fake_dream_redis.expire(_LOCK_KEY, BATCH_LOCK_TTL_SECONDS)
    fake_dream_db.rows["p1"]["lease_expires_at"] = _after(BATCH_LOCK_TTL_SECONDS)


def _assert_marked(fake_dream_db: FakeDreamDb, status: DreamPassStatus) -> None:
    row = fake_dream_db.rows["p1"]
    assert row["status"] is status
    assert row["cleanup_pending_at"] is not None
    assert row["lease_token"] == "tok"


def _assert_cleaned_up(
    fake_dream_db: FakeDreamDb, fake_dream_redis: FakeAsyncRedis, charges: AsyncMock
) -> None:
    assert _charged(charges) == ["consolidate"]
    assert _LOCK_KEY not in fake_dream_redis.store
    assert state_key("p1") not in fake_dream_redis.hashes
    assert input_bundle_key("p1") not in fake_dream_redis.store
    row = fake_dream_db.rows["p1"]
    assert (row["cleanup_pending_at"], row["lease_token"]) == (None, None)


async def _walk(entry) -> None:
    await executor._walk_entry(
        entry, now=datetime.now(timezone.utc), api_key_for=lambda _: "k"
    )


def _after(seconds: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)
