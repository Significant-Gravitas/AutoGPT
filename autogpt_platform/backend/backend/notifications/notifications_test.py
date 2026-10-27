"""Guards on the two scheduled passes, and who product mail may reach.

The scheduler fires these on a fixed interval and the RPC returns as soon as
the task is spawned, so a pass that outlives its tick would otherwise overlap
the next one. Both passes queue their email *before* marking the rows that
suppress a resend, so an overlap is a duplicate email, not just wasted work.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType, SubscriptionTier

from backend.data import product_mail_audience
from backend.data.product_mail_audience import (
    ProductMailStanding,
    product_mail_audience_where,
)
from backend.notifications import notifications as delivery
from backend.notifications.conftest import make_db_client
from backend.notifications.notifications import NotificationManager
from backend.notifications.preferences import SERVICE_MESSAGES
from backend.util.settings import Config, UserNotificationsAudience


@pytest.fixture(scope="session")
def server() -> None:
    return None


def _manager() -> NotificationManager:
    """A manager with no service wiring: `_spawn_pass` touches only `_passes`."""
    manager = NotificationManager.__new__(NotificationManager)
    manager._passes = {}
    return manager


@pytest.mark.asyncio
async def test_a_slow_pass_is_not_started_twice():
    manager = _manager()
    started = 0
    release = asyncio.Event()

    async def slow_pass() -> None:
        nonlocal started
        started += 1
        await release.wait()

    manager._spawn_pass("flush", slow_pass)
    await asyncio.sleep(0)
    manager._spawn_pass("flush", slow_pass)  # the next tick, while still running
    await asyncio.sleep(0)

    assert started == 1, "the second tick must not start an overlapping pass"

    release.set()
    await asyncio.gather(*list(manager._passes.values()))


@pytest.mark.asyncio
async def test_the_next_tick_runs_once_the_previous_pass_finishes():
    manager = _manager()
    started = 0

    async def quick_pass() -> None:
        nonlocal started
        started += 1

    manager._spawn_pass("flush", quick_pass)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    manager._spawn_pass("flush", quick_pass)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert started == 2
    assert not manager._passes, "a finished pass must not stay registered"


@pytest.mark.asyncio
async def test_the_two_passes_do_not_block_each_other():
    manager = _manager()
    ran: list[str] = []
    release = asyncio.Event()

    async def briefings() -> None:
        ran.append("briefings")
        await release.wait()

    async def alerts() -> None:
        ran.append("alerts")

    manager._spawn_pass("send_due_briefings", briefings)
    await asyncio.sleep(0)
    manager._spawn_pass("flush_matured_alerts", alerts)
    await asyncio.sleep(0)

    assert ran == ["briefings", "alerts"]

    release.set()
    await asyncio.gather(*list(manager._passes.values()))


@pytest.mark.asyncio
async def test_a_failing_pass_does_not_wedge_the_schedule():
    """A pass that raises must still clear its slot, or the guard would latch
    on and silently stop every later tick."""
    manager = _manager()

    async def boom() -> None:
        raise RuntimeError("boom")

    manager._spawn_pass("flush", boom)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert not manager._passes

    ran = False

    async def after() -> None:
        nonlocal ran
        ran = True

    manager._spawn_pass("flush", after)
    await asyncio.sleep(0)
    assert ran


@pytest.mark.asyncio
async def test_a_late_done_callback_does_not_clear_a_running_successor():
    """Done callbacks fire via `call_soon`, so a task that finished just before
    the next tick can have its callback run after the successor registers.
    Clearing by name alone would drop the guard while the successor still runs.
    """
    manager = _manager()
    release = asyncio.Event()
    started = 0

    async def first() -> None:
        return None

    async def second() -> None:
        nonlocal started
        started += 1
        await release.wait()

    manager._spawn_pass("flush", first)
    first_task = manager._passes["flush"]
    await asyncio.sleep(0)
    assert first_task.done()

    # Successor registers before the finished task's callback is dispatched.
    manager._spawn_pass("flush", second)
    second_task = manager._passes["flush"]
    assert second_task is not first_task

    await asyncio.sleep(0)
    assert manager._passes.get("flush") is second_task, "successor was evicted"

    manager._spawn_pass("flush", second)
    await asyncio.sleep(0)
    assert started == 1, "guard must still block while the successor runs"

    release.set()
    await second_task


# ── who product mail reaches ────────────────────────────────────────────


def _standing(
    tier: SubscriptionTier = SubscriptionTier.NO_TIER,
    *,
    email: str = "sam@example.com",
    trial_ended_days_ago: int | None = None,
    plan_ended_days_ago: int | None = None,
) -> ProductMailStanding:
    now = datetime.now(timezone.utc)
    return ProductMailStanding(
        email=email,
        subscription_tier=tier,
        subscription_ended_at=(
            now - timedelta(days=plan_ended_days_ago)
            if plan_ended_days_ago is not None
            else None
        ),
        trial_ends_at=(
            now - timedelta(days=trial_ended_days_ago)
            if trial_ended_days_ago is not None
            else None
        ),
    )


@pytest.fixture
def audience(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The switch on, the subscribers audience and no allowlist, unless a test
    changes them through the returned config."""
    monkeypatch.setattr(delivery.settings.config, "enable_user_notifications", True)
    config = product_mail_audience.settings.config
    monkeypatch.setattr(
        config, "user_notifications_audience", UserNotificationsAudience.SUBSCRIBERS
    )
    monkeypatch.setattr(config, "user_notifications_allowlist", "")
    return config


async def _deliver(
    notification_type: NotificationType, standing: ProductMailStanding | None
) -> SimpleNamespace:
    """Run one notification through the user consumer, with everything but the
    audience saying "send it"."""
    sender = AsyncMock()
    manager = NotificationManager.__new__(NotificationManager)
    manager.email_sender = MagicMock(send_notification=sender)
    db_client = make_db_client(
        get_product_mail_standing=AsyncMock(return_value=standing)
    )
    event = SimpleNamespace(type=notification_type, user_id="user-1", data=MagicMock())
    with (
        patch.object(manager, "_parse_message", return_value=event),
        patch.object(
            delivery, "get_database_manager_async_client", return_value=db_client
        ),
        patch.object(delivery, "TrialUpdateData"),
        patch.object(
            delivery, "trial_notice_disposition", AsyncMock(return_value="current")
        ),
        patch.object(delivery, "claim_daily_send", AsyncMock(return_value=True)) as cap,
        patch.object(delivery, "greeting_name", AsyncMock(return_value="Sam")),
        patch.object(delivery, "generate_unsubscribe_link", return_value="u"),
        patch.object(delivery, "generate_preference_link", return_value="p"),
    ):
        handled = await manager._process_user_notification("{}")
    return SimpleNamespace(
        handled=handled, sent=sender.await_count == 1, cap=cap, db=db_client
    )


PRODUCT_MAIL = [NotificationType.BRIEFING, NotificationType.ALERT]


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_type", PRODUCT_MAIL)
@pytest.mark.parametrize(
    "account, standing, gets_it",
    [
        ("free account with runs", _standing(), False),
        ("trialist", _standing(SubscriptionTier.TRIAL), True),
        ("paying account", _standing(SubscriptionTier.PRO), True),
        ("trial ended 3 days ago", _standing(trial_ended_days_ago=3), True),
        ("trial ended 10 days ago", _standing(trial_ended_days_ago=10), False),
        ("plan ended 3 days ago", _standing(plan_ended_days_ago=3), True),
        ("plan ended 10 days ago", _standing(plan_ended_days_ago=10), False),
    ],
)
async def test_product_mail_goes_to_subscribers_trialists_and_the_recently_lapsed(
    audience, notification_type, account, standing, gets_it
):
    outcome = await _deliver(notification_type, standing)

    assert outcome.sent is gets_it, account
    assert outcome.handled is True


@pytest.mark.asyncio
async def test_a_send_outside_the_audience_is_acked_logged_and_claims_no_slot(
    audience, caplog
):
    with caplog.at_level(logging.INFO, logger=delivery.__name__):
        outcome = await _deliver(NotificationType.BRIEFING, _standing())

    # Acked, not retried: a redelivery would only be refused again.
    assert outcome.handled is True
    assert not outcome.sent
    outcome.cap.assert_not_awaited()
    assert "Skipping BRIEFING for user user-1" in caplog.text
    assert "no plan or trial, and none ended in the last 7 days" in caplog.text


@pytest.mark.asyncio
async def test_store_verdicts_are_product_mail_too(audience):
    assert not (await _deliver(NotificationType.VERDICT, _standing())).sent
    assert (
        await _deliver(NotificationType.VERDICT, _standing(SubscriptionTier.PRO))
    ).sent


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_type", sorted(SERVICE_MESSAGES))
async def test_service_mail_reaches_a_free_account(audience, notification_type):
    """A payment-failed notice must get through whatever the audience."""
    outcome = await _deliver(notification_type, _standing())

    assert outcome.sent
    outcome.db.get_product_mail_standing.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_account_that_is_gone_gets_no_product_mail(audience, caplog):
    with caplog.at_level(logging.INFO, logger=delivery.__name__):
        assert not (await _deliver(NotificationType.ALERT, None)).sent
    assert "the account no longer exists" in caplog.text


@pytest.mark.asyncio
async def test_everyone_widens_the_audience_to_free_accounts(audience):
    audience.user_notifications_audience = UserNotificationsAudience.EVERYONE

    assert (await _deliver(NotificationType.BRIEFING, _standing())).sent


@pytest.mark.asyncio
async def test_the_allowlist_alone_decides_while_it_is_set(audience, caplog):
    """A canary day with team inboxes: only the addresses on the list, paying
    or not, and in any case."""
    audience.user_notifications_allowlist = " Team@AGPT.co , qa@agpt.co,"
    team = _standing(email="team@agpt.co")
    customer = _standing(SubscriptionTier.PRO, email="customer@example.com")

    assert (await _deliver(NotificationType.BRIEFING, team)).sent
    with caplog.at_level(logging.INFO, logger=delivery.__name__):
        assert not (await _deliver(NotificationType.BRIEFING, customer)).sent
    assert "not on USER_NOTIFICATIONS_ALLOWLIST" in caplog.text


@pytest.mark.asyncio
async def test_an_allowlist_naming_no_address_lets_no_one_through(audience):
    """A stray comma is a set allowlist, not an unset one: it must not widen a
    canary to the whole audience."""
    audience.user_notifications_allowlist = " , "
    paying = _standing(SubscriptionTier.PRO)

    assert not (await _deliver(NotificationType.BRIEFING, paying)).sent
    assert product_mail_audience_where(datetime.now(timezone.utc)) == {"id": {"in": []}}


@pytest.mark.asyncio
async def test_the_allowlist_overrides_everyone(audience):
    audience.user_notifications_audience = UserNotificationsAudience.EVERYONE
    audience.user_notifications_allowlist = "team@agpt.co"

    assert not (await _deliver(NotificationType.ALERT, _standing())).sent


def test_the_audience_defaults_to_subscribers_and_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("USER_NOTIFICATIONS_AUDIENCE", raising=False)
    monkeypatch.delenv("USER_NOTIFICATIONS_ALLOWLIST", raising=False)
    defaults = Config(_env_file=None)
    assert defaults.user_notifications_audience == UserNotificationsAudience.SUBSCRIBERS
    assert defaults.user_notifications_allowlist == ""
    monkeypatch.setenv("USER_NOTIFICATIONS_AUDIENCE", "Everyone")
    monkeypatch.setenv("USER_NOTIFICATIONS_ALLOWLIST", "a@agpt.co,b@agpt.co")
    config = Config(_env_file=None)
    assert config.user_notifications_audience == UserNotificationsAudience.EVERYONE
    assert config.user_notifications_allowlist == "a@agpt.co,b@agpt.co"


def test_a_misspelt_audience_falls_back_to_subscribers(
    monkeypatch: pytest.MonkeyPatch,
):
    """Boots, on the narrower audience, rather than crashing every process."""
    monkeypatch.setenv("USER_NOTIFICATIONS_AUDIENCE", "everybody")
    assert (
        Config(_env_file=None).user_notifications_audience
        == UserNotificationsAudience.SUBSCRIBERS
    )


def test_the_briefing_pass_selects_by_the_same_rule(audience):
    """`get_briefing_candidates` filters with this clause so the hourly pass
    doesn't queue work the consumer will drop. Its behaviour against a real
    table is pinned in `backend/data/product_mail_audience_test.py`."""
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=7)

    assert product_mail_audience_where(now) == {
        "OR": [
            {"subscriptionTier": {"not": SubscriptionTier.NO_TIER}},
            {"subscriptionEndedAt": {"gte": since}},
            {"subscriptionTrial": {"is": {"endsAt": {"gte": since, "lte": now}}}},
        ]
    }
    audience.user_notifications_audience = UserNotificationsAudience.EVERYONE
    assert product_mail_audience_where(now) is None
    audience.user_notifications_allowlist = "Team@agpt.co"
    assert product_mail_audience_where(now) == {
        "OR": [{"email": {"equals": "team@agpt.co", "mode": "insensitive"}}]
    }
