"""The batch route's handoff meeting a stop, through the sync entry point with
the real dream lock over the in-memory Redis and the pass's row in the
in-memory store: a pass cancelled while it gathered submits nothing, and one
cancelled while it submitted takes its batch back and releases its lock
instead of handing both to callbacks that would only stop."""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import DreamPassStatus

from backend.copilot.graphiti.scope import MemoryScope

from . import batch_handoff as batch_handoff_mod
from . import orchestrator as orchestrator_mod
from .batch_submit import input_bundle_key
from .cancel import cancel_dream_pass
from .fetch import DreamInput, EpisodeRow
from .locks import BATCH_LOCK_TTL_SECONDS

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
        "cancelled: testing",
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
    assert result.error == "cancelled: testing"
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


async def test_a_submit_write_that_fails_on_an_open_row_still_hands_off(
    fake_dream_db, fake_dream_redis, gather, provider, mocker, caplog
):
    record = fake_dream_db.update_dream_pass

    async def submit_write_fails(pass_id, update):
        if update.status is DreamPassStatus.SUBMITTED:
            raise ConnectionError("dream pass database unreachable")
        return await record(pass_id, update)

    mocker.patch.object(fake_dream_db, "update_dream_pass", submit_write_fails)

    with caplog.at_level(logging.WARNING, logger=batch_handoff_mod.logger.name):
        result = await orchestrator_mod.execute_dream_pass("u")

    assert result.error is None
    assert "batch batch-1 is not on its record" in caplog.text
    assert fake_dream_redis.ttls[_LOCK_KEY] == BATCH_LOCK_TTL_SECONDS
    provider.cancel.assert_not_awaited()


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


async def test_a_pass_whose_row_is_gone_is_not_handed_off(
    fake_dream_db, fake_dream_redis, gather, provider, mocker
):
    """A row that was never inserted (or was deleted with its user or
    expert) refuses the submit too: the pass has nothing that could stop it
    later, so it stops here."""
    mocker.patch.object(
        fake_dream_db,
        "create_dream_pass",
        AsyncMock(side_effect=ConnectionError("dream pass database unreachable")),
    )

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
        "testing",
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
