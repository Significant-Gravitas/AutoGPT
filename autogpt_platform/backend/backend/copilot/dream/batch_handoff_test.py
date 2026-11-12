"""The batch route's handoff meeting a stop, through the sync entry point with
the real dream lock over the in-memory Redis and the pass's row in the
in-memory store: a pass cancelled while it gathered submits nothing, and one
cancelled while it submitted takes its batch back and releases its lock
instead of handing both to callbacks that would only stop."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import DreamPassStatus

from backend.copilot.graphiti.scope import MemoryScope
from backend.executor.batch_executor import PendingEntry

from . import batch_handoff as batch_handoff_mod
from . import orchestrator as orchestrator_mod
from . import reaper as reaper_mod
from .batch_deliveries import drop_closed
from .batch_submit import input_bundle_key
from .cancel import cancel_dream_pass
from .fetch import DreamInput, EpisodeRow
from .locks import BATCH_LOCK_TTL_SECONDS
from .provider_batch import cancel_provider_batch
from .reaper import REAP_GRACE_SECONDS, reap_expired_passes

_SCOPE = MemoryScope.for_user("u")
_LOCK_KEY = _SCOPE.redis_key("dream_lock")


@pytest.fixture
def gather(mocker) -> AsyncMock:
    """A pass on the batch route, everything up to the provider stubbed; its
    gather returned so a test can cancel the pass while it gathers."""
    mocker.patch.object(
        orchestrator_mod, "resolve_dream_execution_path", return_value="anthropic_batch"
    )
    config = MagicMock()
    config.direct_anthropic_api_key = "key"
    mocker.patch.object(orchestrator_mod, "ChatConfig", return_value=config)
    mocker.patch.object(
        orchestrator_mod, "is_feature_enabled", AsyncMock(return_value=False)
    )
    mocker.patch.object(
        orchestrator_mod, "check_dream_budget", AsyncMock(return_value=(True, None))
    )
    mocker.patch.object(batch_handoff_mod, "phase_models_for_config", return_value={})
    return mocker.patch.object(
        orchestrator_mod, "gather_dream_input", AsyncMock(return_value=_input())
    )


@pytest.fixture
def provider(mocker) -> MagicMock:
    """The provider side: the submit, the executor's queue and the cancel."""
    sides = MagicMock()
    sides.submit = mocker.patch.object(
        batch_handoff_mod,
        "submit_phase",
        AsyncMock(return_value=MagicMock(provider_batch_id="batch-1")),
    )
    sides.revoke = mocker.patch.object(batch_handoff_mod, "remove_pending", AsyncMock())
    sides.cancel = mocker.patch.object(
        batch_handoff_mod, "cancel_provider_batch", AsyncMock()
    )
    return sides


async def test_a_pass_cancelled_while_it_gathered_submits_nothing(
    fake_dream_db, fake_dream_redis, gather, provider
):
    async def cancelled_gather(scope: MemoryScope) -> DreamInput:
        await _cancel_the_pass(fake_dream_db)
        return _input()

    gather.side_effect = cancelled_gather

    result = await orchestrator_mod.execute_dream_pass("u")

    provider.submit.assert_not_awaited()
    assert (result.error, result.execution_path) == (
        "cancelled: by_admin",
        "anthropic_batch",
    )
    assert input_bundle_key(result.pass_id) not in fake_dream_redis.store
    assert _LOCK_KEY not in fake_dream_redis.store
    _assert_cancelled(fake_dream_db, result.pass_id)


async def test_a_pass_cancelled_while_it_submitted_takes_its_batch_back(
    fake_dream_db, fake_dream_redis, gather, provider
):
    """The batch was submitted before the cancel reached the row, which then
    refuses the submit: the batch is revoked from the executor and cancelled
    at the provider, the bundle dropped, the lock released on the way out."""

    async def submitted_then_cancelled(**_kwargs) -> MagicMock:
        await _cancel_the_pass(fake_dream_db)
        return MagicMock(provider_batch_id="batch-1")

    provider.submit.side_effect = submitted_then_cancelled

    result = await orchestrator_mod.execute_dream_pass("u")

    provider.submit.assert_awaited_once()
    provider.revoke.assert_awaited_once_with("batch-1")
    provider.cancel.assert_awaited_once_with("batch-1")
    assert result.error == "cancelled: by_admin"
    assert input_bundle_key(result.pass_id) not in fake_dream_redis.store
    assert _LOCK_KEY not in fake_dream_redis.store
    _assert_cancelled(fake_dream_db, result.pass_id)
    assert fake_dream_db.rows[result.pass_id].get("provider_batch_id") is None


async def test_a_pass_nobody_stopped_records_its_batch_then_hands_off_its_lock(
    fake_dream_db, fake_dream_redis, gather, provider
):
    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None and not result.skipped
    row = fake_dream_db.rows[result.pass_id]
    assert (row["status"], row["provider_batch_id"]) == (
        DreamPassStatus.SUBMITTED,
        "batch-1",
    )
    assert fake_dream_redis.ttls[_LOCK_KEY] == BATCH_LOCK_TTL_SECONDS
    provider.revoke.assert_not_awaited()
    provider.cancel.assert_not_awaited()


def _submit_write_fails(mocker, fake_dream_db, *, lands: bool) -> None:
    """The submit's write fails as the pass sees it; with *lands* it reached
    the row all the same (a write that committed after the store gave up)."""
    record = fake_dream_db.update_dream_pass

    async def submit_write_fails(pass_id, update):
        if update.status is not DreamPassStatus.SUBMITTED:
            return await record(pass_id, update)
        if lands:
            await record(pass_id, update)
        raise ConnectionError("dream pass database unreachable")

    mocker.patch.object(fake_dream_db, "update_dream_pass", submit_write_fails)


async def test_a_failed_submit_write_whose_batch_the_row_holds_hands_off(
    fake_dream_db, fake_dream_redis, gather, provider, mocker, caplog
):
    """The write landed after all: a read of the row finds the batch on it,
    with its day-long lease, so the pass hands off as usual."""
    _submit_write_fails(mocker, fake_dream_db, lands=True)

    with caplog.at_level(logging.INFO, logger=batch_handoff_mod.logger.name):
        result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    assert "its submit write landed after all" in caplog.text
    row = fake_dream_db.rows[result.pass_id]
    assert (row["status"], row["provider_batch_id"]) == (
        DreamPassStatus.SUBMITTED,
        "batch-1",
    )
    assert fake_dream_redis.ttls[_LOCK_KEY] == BATCH_LOCK_TTL_SECONDS
    provider.revoke.assert_not_awaited()
    provider.cancel.assert_not_awaited()


async def test_a_failed_submit_write_whose_batch_the_row_lacks_takes_it_back(
    fake_dream_db, fake_dream_redis, gather, provider, mocker
):
    """The row does not hold the batch, so nothing would renew the pass's
    half-hour lease while the batch ran, and the reaper would close the live
    pass: the pass takes its batch back (off the executor's queue, cancelled
    at the provider) and ends now, the lock released on the way out."""
    _submit_write_fails(mocker, fake_dream_db, lands=False)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error == batch_handoff_mod.UNRECORDED_SUBMIT_ERROR
    _assert_revoked(provider, fake_dream_redis, result.pass_id)
    assert fake_dream_db.rows[result.pass_id]["status"] is DreamPassStatus.ERRORED


async def test_a_failed_submit_write_whose_row_cannot_be_read_takes_it_back(
    fake_dream_db, fake_dream_redis, gather, provider, mocker
):
    """The read that would find the batch fails too: the pass cannot know
    its lease is held, so it takes its batch back, even though the write
    landed here (the cost of not knowing)."""
    _submit_write_fails(mocker, fake_dream_db, lands=True)
    read = fake_dream_db.get_dream_pass
    reads_after_the_submit = 0

    async def read_fails_after_the_submit(pass_id):
        nonlocal reads_after_the_submit
        if fake_dream_db.rows[pass_id]["status"] is DreamPassStatus.SUBMITTED:
            reads_after_the_submit += 1
            raise ConnectionError("dream pass database unreachable")
        return await read(pass_id)

    mocker.patch.object(fake_dream_db, "get_dream_pass", read_fails_after_the_submit)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert reads_after_the_submit >= 1
    assert result.error == batch_handoff_mod.UNRECORDED_SUBMIT_ERROR
    _assert_revoked(provider, fake_dream_redis, result.pass_id)


async def test_a_refused_submit_whose_row_cannot_be_read_is_not_handed_off(
    fake_dream_db, fake_dream_redis, gather, provider, mocker, caplog
):
    """The row refused the submit (a cancel closed it meanwhile) and the
    read that would say why fails: a refusal is authoritative, so the pass
    stops as if the row had said so, and the reason it could not read is
    logged."""

    async def submitted_then_cancelled_unreadably(**_kwargs) -> MagicMock:
        await _cancel_the_pass(fake_dream_db)
        mocker.patch.object(
            fake_dream_db,
            "get_dream_pass",
            AsyncMock(side_effect=ConnectionError("dream pass database unreachable")),
        )
        return MagicMock(provider_batch_id="batch-1")

    provider.submit.side_effect = submitted_then_cancelled_unreadably

    with caplog.at_level(logging.WARNING, logger=batch_handoff_mod.logger.name):
        result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error == batch_handoff_mod.REFUSED_SUBMIT_ERROR
    assert "refused batch batch-1 and does not say why" in caplog.text
    _assert_revoked(provider, fake_dream_redis, result.pass_id)
    _assert_cancelled(fake_dream_db, result.pass_id)


async def test_a_pass_whose_row_was_never_inserted_submits_nothing(
    fake_dream_db, fake_dream_redis, gather, provider, mocker
):
    """A pass whose insert failed stops before it gathers: a hard forget
    could not find it to mark it, so it reads nothing and submits nothing."""
    mocker.patch.object(
        fake_dream_db,
        "create_dream_pass",
        AsyncMock(side_effect=ConnectionError("dream pass database unreachable")),
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error == "pass: record_unconfirmed"
    gather.assert_not_awaited()
    provider.submit.assert_not_awaited()
    assert _LOCK_KEY not in fake_dream_redis.store


async def test_a_pass_whose_row_was_deleted_is_not_handed_off(
    fake_dream_db, fake_dream_redis, gather, provider
):
    """A row deleted with its user or expert after the gather refuses the
    submit: the pass has nothing that could stop it later, so it stops
    here."""

    async def submitted_after_the_row_went(**_kwargs) -> MagicMock:
        fake_dream_db.rows.clear()
        return MagicMock(provider_batch_id="batch-1")

    provider.submit.side_effect = submitted_after_the_row_went

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error == batch_handoff_mod.REFUSED_SUBMIT_ERROR
    _assert_revoked(provider, fake_dream_redis, result.pass_id)


def _assert_revoked(provider: MagicMock, fake_dream_redis, pass_id: str) -> None:
    """The batch taken back: off the executor's queue and cancelled at the
    provider, the bundle dropped, the lock released on the way out."""
    provider.revoke.assert_awaited_once_with("batch-1")
    provider.cancel.assert_awaited_once_with("batch-1")
    assert input_bundle_key(pass_id) not in fake_dream_redis.store
    assert _LOCK_KEY not in fake_dream_redis.store


async def _cancel_the_pass(fake_dream_db) -> None:
    [pass_id] = list(fake_dream_db.rows)
    assert (await cancel_dream_pass(pass_id, user_id="u", reason="testing")).cancelled


def _assert_cancelled(fake_dream_db, pass_id: str) -> None:
    row = fake_dream_db.rows[pass_id]
    assert (row["status"], row["error"], row["cancel_generation"]) == (
        DreamPassStatus.CANCELLED,
        "cancelled: by_admin",
        1,
    )


def _input() -> DreamInput:
    now = datetime.now(timezone.utc)
    episode = EpisodeRow(
        uuid="e1",
        name=None,
        content="hello",
        source_description=None,
        valid_at=None,
        created_at=None,
    )
    return DreamInput(
        user_id="u",
        group_id=_SCOPE.group_id,
        window_start=now - timedelta(days=14),
        window_end=now,
        episodes=[episode],
    )


def _the_stop_is_not_confirmed(mocker, provider, how: str) -> None:
    """The provider does not confirm the take-back's cancel: it refuses it
    (``denied``), or the call outlasts its deadline (``timed_out``, through
    the real cancel)."""
    if how == "denied":
        provider.cancel.return_value = False
        return

    async def hangs(**_kwargs) -> bool:
        await asyncio.Event().wait()
        return True

    mocker.patch.object(
        batch_handoff_mod, "cancel_provider_batch", cancel_provider_batch
    )
    mocker.patch(
        "backend.copilot.dream.provider_batch.anthropic_api_key", return_value="k"
    )
    mocker.patch(
        "backend.copilot.dream.provider_batch.PROVIDER_CANCEL_TIMEOUT_SECONDS", 0.05
    )
    provider.cancel = mocker.patch(
        "backend.copilot.dream.provider_batch.cancel_batch",
        AsyncMock(side_effect=hangs),
    )


def _submit_lands_during_the_cancel(mocker, fake_dream_db, provider) -> None:
    """The submit's write fails as the pass sees it and lands later, after
    the read that looked for it: while the take-back cancels the batch."""
    record = fake_dream_db.update_dream_pass
    late: list = []

    async def lands_late(pass_id, update):
        if update.status is DreamPassStatus.SUBMITTED:
            late.append((pass_id, update))
            raise ConnectionError("dream pass database unreachable")
        return await record(pass_id, update)

    mocker.patch.object(fake_dream_db, "update_dream_pass", lands_late)
    cancel = provider.cancel.side_effect

    async def cancel_after_the_write_lands(*args, **kwargs):
        for pass_id, update in late:
            await record(pass_id, update)
        if cancel is not None:
            return await cancel(*args, **kwargs)
        return provider.cancel.return_value

    provider.cancel.side_effect = cancel_after_the_write_lands


@pytest.mark.parametrize("stop", ["denied", "timed_out"])
@pytest.mark.parametrize("submit", ["absent", "landed_late"])
async def test_a_batch_taken_back_without_a_confirmed_stop_stays_on_its_record(
    fake_dream_db, fake_dream_redis, gather, provider, mocker, stop, submit
):
    """The take-back cannot confirm the batch stopped: the row closes
    ERRORED with the batch on it and marked, so the reaper's cleanup keeps
    asking the provider to stop it; the batch leaves the executor's queue,
    the bundle goes and the lock is released once, on the way out."""
    _the_stop_is_not_confirmed(mocker, provider, stop)
    if submit == "absent":
        _submit_write_fails(mocker, fake_dream_db, lands=False)
    else:
        _submit_lands_during_the_cancel(mocker, fake_dream_db, provider)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error == batch_handoff_mod.UNRECORDED_SUBMIT_ERROR
    provider.cancel.assert_awaited_once()
    row = fake_dream_db.rows[result.pass_id]
    assert (row["status"], row["provider_batch_id"]) == (
        DreamPassStatus.ERRORED,
        "batch-1",
    )
    assert row["cleanup_pending_at"] is not None
    provider.revoke.assert_awaited_once_with("batch-1")
    assert input_bundle_key(result.pass_id) not in fake_dream_redis.store
    assert _LOCK_KEY not in fake_dream_redis.store


async def test_a_batch_neither_stopped_nor_recorded_stays_on_the_executors_queue(
    fake_dream_db, fake_dream_redis, gather, provider, mocker
):
    """The stop is not confirmed and the row will not take the batch either
    (the store down for writes): the batch stays on the executor's queue,
    whose drop or callback stops it by its id; the bundle goes and the lock
    is released on the way out."""
    _the_stop_is_not_confirmed(mocker, provider, "denied")
    record = fake_dream_db.update_dream_pass

    async def store_down_from_the_submit(pass_id, update):
        if update.status in (DreamPassStatus.SUBMITTED, DreamPassStatus.ERRORED):
            raise ConnectionError("dream pass database unreachable")
        return await record(pass_id, update)

    mocker.patch.object(fake_dream_db, "update_dream_pass", store_down_from_the_submit)

    result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error == batch_handoff_mod.UNRECORDED_SUBMIT_ERROR
    provider.cancel.assert_awaited_once_with("batch-1")
    provider.revoke.assert_not_awaited()
    assert input_bundle_key(result.pass_id) not in fake_dream_redis.store
    assert _LOCK_KEY not in fake_dream_redis.store


async def test_a_take_back_racing_a_cancel_leaves_the_batch_to_its_expiry(
    fake_dream_db, fake_dream_redis, gather, provider, mocker
):
    """The known limit at ``_take_back``: a cancel closes the row while the
    batch is submitted, and the provider confirms neither the take-back's
    stop nor the executor's drop's. The closed row does not take the batch,
    so nothing asks again and the batch is left to the provider's expiry:
    nothing raises, nothing is charged, and the row stays how the cancel
    closed it."""

    async def submitted_then_cancelled(**_kwargs) -> MagicMock:
        await _cancel_the_pass(fake_dream_db)
        return MagicMock(provider_batch_id="batch-1")

    provider.submit.side_effect = submitted_then_cancelled
    provider.cancel.return_value = False
    stop = mocker.patch(
        "backend.copilot.dream.cleanup.provider_batch_stopped",
        AsyncMock(return_value=False),
    )
    charge = mocker.patch(
        "backend.copilot.dream.batch_costs.record_phase_cost", AsyncMock()
    )

    result = await orchestrator_mod.execute_dream_pass("u")

    _assert_cancelled(fake_dream_db, result.pass_id)
    assert fake_dream_db.rows[result.pass_id].get("provider_batch_id") is None

    await drop_closed(_batch_entry(result.pass_id))
    asked = stop.await_count
    mocker.patch.object(reaper_mod, "phase_models_for_config", return_value={})
    await reap_expired_passes(
        now=datetime.now(timezone.utc) + timedelta(seconds=REAP_GRACE_SECONDS + 60)
    )

    _assert_cancelled(fake_dream_db, result.pass_id)
    assert fake_dream_db.rows[result.pass_id].get("provider_batch_id") is None
    assert stop.await_count == asked, "nothing asks the provider again"
    charge.assert_not_awaited()


def _batch_entry(pass_id: str) -> PendingEntry:
    """The executor's entry for the pass's consolidate batch."""
    now = datetime.now(timezone.utc)
    return PendingEntry(
        provider="anthropic",
        provider_batch_id="batch-1",
        callback_namespace="dream_pass",
        submitted_at=now,
        next_poll_at=now,
        payload={
            "user_id": "u",
            "cost_context": {"user_id": "u", "trial_id": None, "generation": None},
            "pass_id": pass_id,
            "job_id": "",
            "phase": "consolidate",
            "phase_models": {},
        },
    )
