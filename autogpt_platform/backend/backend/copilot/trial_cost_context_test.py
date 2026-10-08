import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.copilot.trial_cost_context import (
    TrialCostContext,
    attributed_usage,
    get_trial_cost_context,
    record_attributed_trial_cost,
    restore_cost_context,
    trial_cost_context,
)
from backend.copilot.usage_activation import UsageActivationUnavailable
from backend.data.pro_activation import UsageActivationState


@pytest.fixture
def store(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        return_value=UsageActivationState(
            user_id="user-1",
            trial_id="trial-1",
            tier=SubscriptionTier.TRIAL,
            ready=True,
        )
    )
    store.record_subscription_trial_cost = AsyncMock()
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )
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
            with restore_cost_context(
                None, TrialCostContext(user_id=None, trial_id=None)
            ):
                raise exit_error()
        assert get_trial_cost_context("user-1") is parent
    assert get_trial_cost_context("user-1") is None
    store.get_usage_activation_state.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_cross_user_cost_is_rejected_before_writes(store):
    async with trial_cost_context("user-1"):
        with pytest.raises(ValueError, match="different user"):
            await record_attributed_trial_cost("user-2", 100)
    store.record_subscription_trial_cost.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_trial_snapshot_never_picks_up_later_enrollment(store):
    store.get_usage_activation_state.return_value.trial_id = None
    async with trial_cost_context("user-1"):
        store.get_usage_activation_state.return_value.trial_id = "trial-later"
        assert await record_attributed_trial_cost("user-1", 100) is True
    store.record_subscription_trial_cost.assert_not_awaited()


@pytest.mark.asyncio
async def test_unscoped_call_does_not_guess_trial_attribution(store):
    assert await record_attributed_trial_cost("user-1", 100) is False
    store.get_usage_activation_state.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [True, False])
async def test_unavailable_state_prevents_work_and_does_not_leak_context(
    store, pending
):
    if pending:
        store.get_usage_activation_state.return_value.ready = False
    else:
        store.get_usage_activation_state.side_effect = ConnectionError("Database down")
    with pytest.raises(UsageActivationUnavailable):
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
async def test_child_and_background_work_keep_pre_activation_generation(store):
    release = asyncio.Event()
    snapshots = []

    @attributed_usage
    async def background(user_id):
        await release.wait()
        snapshots.append(get_trial_cost_context(user_id))
        await record_attributed_trial_cost(user_id, 100)

    async with trial_cost_context("user-1"):
        child = asyncio.create_task(background("user-1"))
    store.get_usage_activation_state.return_value = UsageActivationState(
        user_id="user-1", generation="paid-1", tier=SubscriptionTier.PRO, ready=True
    )
    release.set()
    await child
    await background("user-1")
    assert [(c.trial_id, c.generation) for c in snapshots] == [
        ("trial-1", None),
        (None, "paid-1"),
    ]
    store.record_subscription_trial_cost.assert_awaited_once_with(
        "user-1", 100, trial_id="trial-1"
    )


@pytest.mark.asyncio
async def test_serialized_child_context_survives_process_boundary(store):
    async with trial_cost_context("user-1"):
        original = get_trial_cost_context("user-1")
        encoded = original.model_dump_json()
    restored = TrialCostContext.model_validate_json(encoded)
    async with trial_cost_context("user-1", restored):
        assert get_trial_cost_context("user-1") == original
    store.get_usage_activation_state.assert_awaited_once()
