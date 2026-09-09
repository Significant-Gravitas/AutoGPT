import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.trial_cost_context import (
    get_trial_cost_context,
    record_attributed_trial_cost,
    trial_cost_context,
)
from backend.data.subscription_trial import TrialState


@pytest.fixture
def store(mocker):
    trial = MagicMock(spec=TrialState)
    trial.id = "trial-1"
    trial.active = True
    trial.consumed_at = datetime.now(UTC)
    store = MagicMock()
    store.get_subscription_trial = AsyncMock(return_value=trial)
    store.record_subscription_trial_cost = AsyncMock()
    mocker.patch(
        "backend.copilot.trial_cost_context.db_accessors.credit_db", return_value=store
    )
    return store


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_error", [RuntimeError, asyncio.CancelledError])
async def test_scope_restores_parent_even_on_failure(store, exit_error):
    assert get_trial_cost_context("user-1") is None
    async with trial_cost_context("user-1"):
        parent = get_trial_cost_context("user-1")
        with pytest.raises(exit_error):
            async with trial_cost_context(None):
                anonymous = get_trial_cost_context(None)
                assert anonymous is not None and anonymous.trial_id is None
                raise exit_error()
        assert get_trial_cost_context("user-1") is parent
    assert get_trial_cost_context("user-1") is None
    store.get_subscription_trial.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_cross_user_cost_is_rejected_before_writes(store):
    async with trial_cost_context("user-1"):
        with pytest.raises(ValueError, match="different user"):
            await record_attributed_trial_cost("user-2", 100)
    store.record_subscription_trial_cost.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_state", ["absent", "inactive", "unconsumed"])
async def test_non_trial_snapshot_never_picks_up_later_enrollment(store, initial_state):
    trial = store.get_subscription_trial.return_value
    if initial_state == "absent":
        store.get_subscription_trial.return_value = None
    elif initial_state == "inactive":
        trial.active = False
    else:
        trial.consumed_at = None
    async with trial_cost_context("user-1"):
        store.get_subscription_trial.return_value = trial
        trial.active = True
        trial.consumed_at = datetime.now(UTC)
        assert await record_attributed_trial_cost("user-1", 100) is True
    store.record_subscription_trial_cost.assert_not_awaited()


@pytest.mark.asyncio
async def test_unscoped_call_preserves_legacy_recording(store):
    assert await record_attributed_trial_cost("user-1", 100) is False
    store.get_subscription_trial.assert_not_awaited()
    store.record_subscription_trial_cost.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_failure_prevents_work_and_does_not_leak_context(store):
    store.get_subscription_trial.side_effect = ConnectionError("Database unavailable")
    with pytest.raises(ConnectionError):
        async with trial_cost_context("user-1"):
            pytest.fail("Work must not start without an attribution snapshot")
    assert get_trial_cost_context("user-1") is None


@pytest.mark.asyncio
async def test_accounting_failure_is_not_silently_discarded(store):
    store.record_subscription_trial_cost.side_effect = ConnectionError("Database down")
    async with trial_cost_context("user-1"):
        with pytest.raises(ConnectionError):
            await record_attributed_trial_cost("user-1", 100)


@pytest.mark.asyncio
async def test_parallel_users_do_not_share_attribution(store):
    async def lookup(user_id):
        trial = MagicMock(spec=TrialState)
        trial.id = f"trial-{user_id}"
        trial.active = True
        trial.consumed_at = datetime.now(UTC)
        return trial

    store.get_subscription_trial.side_effect = lookup
    barrier = asyncio.Barrier(2)

    async def turn(user_id):
        async with trial_cost_context(user_id):
            await barrier.wait()
            assert await record_attributed_trial_cost(user_id, 100) is True

    await asyncio.gather(turn("a"), turn("b"))
    assert store.record_subscription_trial_cost.await_count == 2
    store.record_subscription_trial_cost.assert_any_await("a", 100, trial_id="trial-a")
    store.record_subscription_trial_cost.assert_any_await("b", 100, trial_id="trial-b")
    assert get_trial_cost_context("a") is None
