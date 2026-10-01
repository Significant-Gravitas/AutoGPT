import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import posthog_lifecycle_sync as lifecycle
from backend.data.posthog_lifecycle import LifecycleSnapshot, LifecycleUser

MODULE = "backend.data.posthog_lifecycle_sync"
SIGNUP = datetime(2026, 1, 1, tzinfo=UTC)
NOW = datetime(2026, 9, 29, tzinfo=UTC)


def stripe_page(*subs: dict, has_more: bool = False) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {"object": "list", "data": list(subs), "has_more": has_more}, "k"
    )


def stripe_sub(sub_id: str, customer: str, status: str = "active", **extra) -> dict:
    start = int((NOW - timedelta(days=10)).timestamp())
    return {
        "id": sub_id,
        "customer": customer,
        "status": status,
        "start_date": start,
        "created": start,
        "metadata": {},
        **extra,
    }


def db_user(customer: str | None = "cus_1", tier=SubscriptionTier.PRO) -> MagicMock:
    return MagicMock(
        id="user-1",
        createdAt=SIGNUP,
        stripeCustomerId=customer,
        subscriptionTier=tier,
        subscriptionTrial=None,
    )


@pytest.fixture
def posthog():
    client = MagicMock()
    with patch(f"{MODULE}.get_posthog_client", return_value=client):
        yield client


@pytest.fixture(autouse=True)
def clean_background_state():
    lifecycle._syncs.clear()
    yield
    lifecycle._syncs.clear()


# ---- send_lifecycle_snapshot ------------------------------------------------


def test_send_sets_and_unsets_on_the_user_id(posthog):
    snapshot = LifecycleSnapshot(subscription_status="signed", signup_at=SIGNUP)

    assert lifecycle.send_lifecycle_snapshot("user-1", snapshot) is True

    posthog.capture.assert_called_once_with(
        "$set",
        distinct_id="user-1",
        properties={
            "$set": {
                "subscription_status": "signed",
                "signup_at": "2026-01-01T00:00:00+00:00",
            },
            "$unset": [
                "trial_started_at",
                "subscription_started_at",
                "subscription_canceled_at",
                "subscription_ended_at",
            ],
        },
    )


def test_send_without_user_id_or_client_is_a_no_op(posthog):
    snapshot = LifecycleSnapshot(subscription_status="signed")
    assert lifecycle.send_lifecycle_snapshot("", snapshot) is False
    posthog.capture.assert_not_called()
    with patch(f"{MODULE}.get_posthog_client", return_value=None):
        assert lifecycle.send_lifecycle_snapshot("user-1", snapshot) is False


def test_send_swallows_posthog_errors(posthog):
    posthog.capture.side_effect = RuntimeError("posthog down")
    snapshot = LifecycleSnapshot(subscription_status="signed")
    assert lifecycle.send_lifecycle_snapshot("user-1", snapshot) is False


# ---- sync_posthog_lifecycle -------------------------------------------------


async def test_sync_reads_the_customers_current_subscriptions(posthog):
    find_unique = AsyncMock(return_value=db_user())
    list_async = AsyncMock(return_value=stripe_page(stripe_sub("sub_1", "cus_1")))
    with (
        patch(f"{MODULE}.User.prisma", return_value=MagicMock(find_unique=find_unique)),
        patch.object(stripe.Subscription, "list_async", list_async),
    ):
        snapshot = await lifecycle.sync_posthog_lifecycle("user-1")

    assert snapshot is not None
    assert snapshot.subscription_status == "subscribed"
    list_async.assert_awaited_once_with(customer="cus_1", status="all", limit=100)
    sent = posthog.capture.call_args.kwargs["properties"]["$set"]
    assert sent["subscription_status"] == "subscribed"
    assert sent["signup_at"] == "2026-01-01T00:00:00+00:00"


async def test_sync_skips_stripe_for_users_without_a_customer(posthog):
    find_unique = AsyncMock(return_value=db_user(customer=None, tier=None))
    list_async = AsyncMock()
    with (
        patch(f"{MODULE}.User.prisma", return_value=MagicMock(find_unique=find_unique)),
        patch.object(stripe.Subscription, "list_async", list_async),
    ):
        snapshot = await lifecycle.sync_posthog_lifecycle("user-1")

    assert snapshot is not None
    assert snapshot.subscription_status == "signed"
    list_async.assert_not_awaited()
    posthog.capture.assert_called_once()


async def test_sync_uses_a_supplied_subscription_list(posthog):
    find_unique = AsyncMock(return_value=db_user())
    list_async = AsyncMock()
    with (
        patch(f"{MODULE}.User.prisma", return_value=MagicMock(find_unique=find_unique)),
        patch.object(stripe.Subscription, "list_async", list_async),
    ):
        snapshot = await lifecycle.sync_posthog_lifecycle(
            "user-1", stripe_subscriptions=[]
        )

    assert snapshot is not None
    list_async.assert_not_awaited()


async def test_sync_is_a_no_op_when_analytics_is_off():
    prisma = MagicMock()
    with (
        patch(f"{MODULE}.get_posthog_client", return_value=None),
        patch(f"{MODULE}.User.prisma", prisma),
    ):
        assert await lifecycle.sync_posthog_lifecycle("user-1") is None
    prisma.assert_not_called()


async def test_sync_swallows_stripe_errors_and_sends_nothing(posthog):
    find_unique = AsyncMock(return_value=db_user())
    with (
        patch(f"{MODULE}.User.prisma", return_value=MagicMock(find_unique=find_unique)),
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(side_effect=stripe.APIConnectionError("down")),
        ),
    ):
        assert await lifecycle.sync_posthog_lifecycle("user-1") is None
    posthog.capture.assert_not_called()


async def test_sync_swallows_posthog_errors(posthog):
    posthog.capture.side_effect = RuntimeError("posthog down")
    find_unique = AsyncMock(return_value=db_user(customer=None))
    with patch(
        f"{MODULE}.User.prisma", return_value=MagicMock(find_unique=find_unique)
    ):
        snapshot = await lifecycle.sync_posthog_lifecycle("user-1")
    assert snapshot is not None


async def test_sync_of_a_deleted_user_sends_nothing(posthog):
    find_unique = AsyncMock(return_value=None)
    with patch(
        f"{MODULE}.User.prisma", return_value=MagicMock(find_unique=find_unique)
    ):
        assert await lifecycle.sync_posthog_lifecycle("user-1") is None
    posthog.capture.assert_not_called()


# ---- schedule_posthog_lifecycle_sync ----------------------------------------


async def _drain() -> None:
    # Done tasks leave the set in a callback, so yield to let it run.
    while lifecycle._background_tasks:
        await asyncio.gather(*list(lifecycle._background_tasks))
        await asyncio.sleep(0)


async def test_schedule_runs_the_sync_in_the_background(posthog):
    with patch(f"{MODULE}.sync_posthog_lifecycle", new_callable=AsyncMock) as sync:
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        sync.assert_not_awaited()
        await _drain()
    sync.assert_awaited_once_with("user-1")


async def test_schedule_coalesces_calls_that_are_still_queued(posthog):
    with patch(f"{MODULE}.sync_posthog_lifecycle", new_callable=AsyncMock) as sync:
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        await _drain()
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        await _drain()
    assert sync.await_count == 2


async def test_schedule_by_customer_resolves_the_user(posthog):
    find_first = AsyncMock(return_value=MagicMock(id="user-9"))
    with (
        patch(f"{MODULE}.User.prisma", return_value=MagicMock(find_first=find_first)),
        patch(f"{MODULE}.sync_posthog_lifecycle", new_callable=AsyncMock) as sync,
    ):
        lifecycle.schedule_posthog_lifecycle_sync(stripe_customer_id="cus_9")
        await _drain()
    find_first.assert_awaited_once_with(where={"stripeCustomerId": "cus_9"})
    sync.assert_awaited_once_with("user-9")


async def test_scheduled_sync_failure_stays_in_the_background(posthog):
    with patch(
        f"{MODULE}.sync_posthog_lifecycle",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        await _drain()


async def test_schedule_is_a_no_op_when_analytics_is_off():
    with (
        patch(f"{MODULE}.get_posthog_client", return_value=None),
        patch(f"{MODULE}.sync_posthog_lifecycle", new_callable=AsyncMock) as sync,
    ):
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
    assert not lifecycle._background_tasks
    sync.assert_not_called()


def test_schedule_without_a_running_loop_does_not_raise(posthog):
    lifecycle.schedule_posthog_lifecycle_sync("user-1")
    assert not lifecycle._syncs


def test_schedule_swallows_client_errors():
    with patch(f"{MODULE}.get_posthog_client", side_effect=RuntimeError("bad key")):
        lifecycle.schedule_posthog_lifecycle_sync("user-1")


# ---- sync_all_posthog_lifecycles --------------------------------------------


def batch_user(user_id: str, customer: str | None, tier: str = "NO_TIER"):
    return LifecycleUser(
        id=user_id,
        created_at=SIGNUP,
        stripe_customer_id=customer,
        subscription_tier=SubscriptionTier(tier),
    )


@pytest.fixture
def sweep_sources():
    """Two DB batches over three users, and every subscription in Stripe."""
    batches = [
        [
            batch_user("u-sub", "cus_sub", "PRO"),
            batch_user("u-canceling", "cus_canceling", "PRO"),
        ],
        [batch_user("u-ended", "cus_ended")],
        [],
    ]
    subscriptions = stripe_page(
        stripe_sub("sub_1", "cus_sub"),
        stripe_sub("sub_2", "cus_canceling", cancel_at_period_end=True),
        stripe_sub("sub_3", "cus_ended", "canceled", ended_at=1_780_000_000),
        stripe_sub("sub_4", "cus_not_ours"),
    )
    list_async = AsyncMock(return_value=subscriptions)
    with (
        patch(
            f"{MODULE}.query_raw_with_schema",
            new_callable=AsyncMock,
            side_effect=batches,
        ) as query,
        patch.object(stripe.Subscription, "list_async", list_async),
        patch(
            f"{MODULE}.SubscriptionTrial.prisma",
            return_value=MagicMock(find_many=AsyncMock(return_value=[])),
        ),
    ):
        yield query, list_async


async def test_sweep_dry_run_counts_and_sends_nothing(sweep_sources):
    query, list_async = sweep_sources
    with patch(f"{MODULE}.get_posthog_client") as get_client:
        summary = await lifecycle.sync_all_posthog_lifecycles(dry_run=True, now=NOW)

    get_client.assert_not_called()
    list_async.assert_awaited_once_with(status="all", limit=100)
    assert summary.aborted is None
    assert summary.stripe_subscriptions == 4
    assert summary.users == 3
    assert summary.sent == 0
    assert summary.status_counts == {
        "subscribed": 1,
        "subscription_canceled": 1,
        "subscription_ended": 1,
    }
    # Default scope: users with billing history only.
    assert query.await_args_list[0].args[1:] == ("", 500, False)
    assert query.await_args_list[1].args[1] == "u-canceling"


async def test_sweep_sends_every_user_and_flushes_per_batch(sweep_sources, posthog):
    summary = await lifecycle.sync_all_posthog_lifecycles(now=NOW)

    assert summary.sent == 3
    assert summary.errors == 0
    sent = {
        call.kwargs["distinct_id"]: call.kwargs["properties"]["$set"][
            "subscription_status"
        ]
        for call in posthog.capture.call_args_list
    }
    assert sent == {
        "u-sub": "subscribed",
        "u-canceling": "subscription_canceled",
        "u-ended": "subscription_ended",
    }
    assert posthog.flush.call_count == 2


async def test_sweep_all_users_widens_the_query(sweep_sources, posthog):
    query, _ = sweep_sources
    await lifecycle.sync_all_posthog_lifecycles(all_users=True, now=NOW)
    assert query.await_args_list[0].args[3] is True


async def test_sweep_aborts_when_stripe_listing_fails(posthog):
    with (
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(side_effect=stripe.APIConnectionError("down")),
        ),
        patch(f"{MODULE}.query_raw_with_schema", new_callable=AsyncMock) as query,
    ):
        summary = await lifecycle.sync_all_posthog_lifecycles(now=NOW)

    assert summary.aborted == "Stripe listing failed"
    query.assert_not_awaited()
    posthog.capture.assert_not_called()


async def test_sweep_aborts_before_stripe_when_posthog_is_off():
    list_async = AsyncMock()
    with (
        patch(f"{MODULE}.get_posthog_client", return_value=None),
        patch.object(stripe.Subscription, "list_async", list_async),
    ):
        summary = await lifecycle.sync_all_posthog_lifecycles(now=NOW)
    assert summary.aborted == "PostHog is not configured"
    list_async.assert_not_awaited()


async def test_an_aborted_sweep_is_logged(caplog):
    aborted = lifecycle.LifecycleSyncSummary(
        dry_run=False, aborted="PostHog is not configured"
    )
    with patch(
        f"{MODULE}.sync_all_posthog_lifecycles", AsyncMock(return_value=aborted)
    ):
        await lifecycle._run_sweep()
    assert "Lifecycle sweep aborted: PostHog is not configured" in caplog.text


async def test_sweep_skips_a_user_whose_trial_row_is_unreadable(posthog):
    broken = MagicMock(userId="u-trial", offer={"not": "an offer"})
    with (
        patch(
            f"{MODULE}.query_raw_with_schema",
            new_callable=AsyncMock,
            side_effect=[[batch_user("u-trial", "cus_t")], []],
        ),
        patch.object(
            stripe.Subscription, "list_async", AsyncMock(return_value=stripe_page())
        ),
        patch(
            f"{MODULE}.SubscriptionTrial.prisma",
            return_value=MagicMock(find_many=AsyncMock(return_value=[broken])),
        ),
    ):
        summary = await lifecycle.sync_all_posthog_lifecycles(now=NOW)

    assert summary.errors == 1
    assert summary.sent == 0
    posthog.capture.assert_not_called()


# ---- start_posthog_lifecycle_sweep ------------------------------------------


async def test_sweep_starter_returns_at_once_and_never_runs_two():
    release = asyncio.Event()

    async def slow_sweep(**_):
        await release.wait()
        return lifecycle.LifecycleSyncSummary(dry_run=False)

    with patch(f"{MODULE}.sync_all_posthog_lifecycles", side_effect=slow_sweep) as run:
        assert await lifecycle.start_posthog_lifecycle_sweep() is True
        assert await lifecycle.start_posthog_lifecycle_sweep() is False
        release.set()
        await asyncio.gather(*list(lifecycle._sweeps))
        assert run.await_count == 1
        assert await lifecycle.start_posthog_lifecycle_sweep() is True
        await asyncio.gather(*list(lifecycle._sweeps))


async def test_sweep_failure_stays_in_the_background():
    with patch(
        f"{MODULE}.sync_all_posthog_lifecycles",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        assert await lifecycle.start_posthog_lifecycle_sweep() is True
        await asyncio.gather(*list(lifecycle._sweeps))


async def test_scheduled_syncs_run_at_most_four_at_a_time(posthog):
    running = 0
    peak = 0
    release = asyncio.Event()

    async def slow_sync(user_id: str):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1

    with patch(f"{MODULE}.sync_posthog_lifecycle", side_effect=slow_sync) as sync:
        for index in range(12):
            lifecycle.schedule_posthog_lifecycle_sync(f"user-{index}")
        for _ in range(5):
            await asyncio.sleep(0)
        assert peak == lifecycle._MAX_CONCURRENT_SYNCS
        release.set()
        await _drain()
    assert sync.await_count == 12
    assert peak == lifecycle._MAX_CONCURRENT_SYNCS


async def test_a_request_during_a_running_sync_runs_once_more_never_overlapping(
    posthog,
):
    """The race: a sync that started before a write must not be the last
    one sent, and two syncs for one user must never overlap."""
    running = 0
    overlap = False
    started = asyncio.Event()
    release = asyncio.Event()
    reads: list[int] = []
    state = {"version": 1}

    async def sync(user_id: str):
        nonlocal running, overlap
        running += 1
        overlap = overlap or running > 1
        reads.append(state["version"])
        started.set()
        await release.wait()
        running -= 1

    with patch(f"{MODULE}.sync_posthog_lifecycle", side_effect=sync) as run:
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        await started.wait()
        state["version"] = 2
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        release.set()
        await _drain()

    assert run.await_count == 2
    assert reads == [1, 2]
    assert not overlap
    assert not lifecycle._syncs


async def test_customer_hooks_are_coordinated_per_user(posthog):
    """A customer id is resolved first, then goes through the same per-user
    queue as a user-id hook, so the two can't overlap for one user."""
    find_first = AsyncMock(return_value=MagicMock(id="user-1"))
    with (
        patch(f"{MODULE}.User.prisma", return_value=MagicMock(find_first=find_first)),
        patch(f"{MODULE}._schedule_user") as schedule_user,
    ):
        lifecycle.schedule_posthog_lifecycle_sync(stripe_customer_id="cus_1")
        await _drain()
    schedule_user.assert_called_once_with("user-1")


async def test_a_sync_cancelled_while_waiting_for_a_slot_does_not_block_the_user(
    posthog,
):
    release = asyncio.Event()
    synced: list[str] = []

    async def sync(user_id: str):
        if user_id.startswith("busy-"):
            await release.wait()
        synced.append(user_id)

    with patch(f"{MODULE}.sync_posthog_lifecycle", side_effect=sync):
        for index in range(lifecycle._MAX_CONCURRENT_SYNCS):
            lifecycle.schedule_posthog_lifecycle_sync(f"busy-{index}")
        await asyncio.sleep(0)
        before = set(lifecycle._background_tasks)
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        (waiting,) = lifecycle._background_tasks - before
        await asyncio.sleep(0)
        assert lifecycle._syncs["user-1"] == "queued"

        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        assert waiting.cancelled()
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        release.set()
        await _drain()

    assert synced.count("user-1") == 1
    assert not lifecycle._syncs


async def test_a_cancelled_running_sync_keeps_a_requested_rerun(posthog):
    started = asyncio.Event()
    calls = 0

    async def sync(user_id: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await asyncio.Event().wait()

    with patch(f"{MODULE}.sync_posthog_lifecycle", side_effect=sync):
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        (running,) = lifecycle._background_tasks
        await started.wait()
        lifecycle.schedule_posthog_lifecycle_sync("user-1")
        assert lifecycle._syncs["user-1"] == "rerun"
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)
        assert running.cancelled()
        await _drain()

    assert calls == 2
    assert not lifecycle._syncs
