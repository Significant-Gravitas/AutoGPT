"""The PostHog lifecycle backfill, end to end against mocked Stripe, database
and PostHog. Nothing here talks to a real account."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

import scripts.backfill_posthog_lifecycle as backfill
from backend.data.posthog_lifecycle import LifecycleUser

SYNC = "backend.data.posthog_lifecycle_sync"
NOW = datetime.now(UTC)


def _sub(sub_id: str, customer: str, status: str, **extra) -> dict:
    start = int((NOW - timedelta(days=30)).timestamp())
    return {
        "id": sub_id,
        "customer": customer,
        "status": status,
        "start_date": start,
        "metadata": {},
        **extra,
    }


def _user(user_id: str, customer: str | None, tier: str) -> LifecycleUser:
    return LifecycleUser(
        id=user_id,
        created_at=NOW - timedelta(days=100),
        stripe_customer_id=customer,
        subscription_tier=SubscriptionTier(tier),
    )


@pytest.fixture
def secrets():
    return MagicMock(
        stripe_api_key="sk_test_unused",
        posthog_api_key="phc_unused",
        posthog_host="https://posthog.invalid",
    )


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch, secrets: MagicMock):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(stripe, "api_key", stripe.api_key)
    posthog = MagicMock()
    subscriptions = stripe.ListObject.construct_from(
        {
            "object": "list",
            "data": [
                _sub("sub_1", "cus_1", "active"),
                _sub("sub_2", "cus_2", "canceled", ended_at=int(NOW.timestamp())),
            ],
            "has_more": False,
        },
        "k",
    )
    users = [
        _user("u-1", "cus_1", "PRO"),
        _user("u-2", "cus_2", "NO_TIER"),
        _user("u-3", "cus_3", "NO_TIER"),
    ]
    with (
        patch(
            "backend.util.settings.Settings", return_value=MagicMock(secrets=secrets)
        ),
        patch("backend.data.db.connect", new_callable=AsyncMock),
        patch("backend.data.db.disconnect", new_callable=AsyncMock),
        patch.object(
            stripe.Subscription, "list_async", AsyncMock(return_value=subscriptions)
        ) as list_async,
        patch(
            f"{SYNC}.query_raw_with_schema",
            new_callable=AsyncMock,
            side_effect=[users, []],
        ),
        patch(
            f"{SYNC}.SubscriptionTrial.prisma",
            return_value=MagicMock(find_many=AsyncMock(return_value=[])),
        ),
        patch(f"{SYNC}.get_posthog_client", return_value=posthog),
        patch("backend.util.posthog_client.get_posthog_client", return_value=posthog),
    ):
        yield posthog, list_async


async def test_dry_run_prints_counts_and_sends_nothing(sources, capsys):
    posthog, list_async = sources

    assert await backfill.main(send=False, all_users=False, batch_size=500) == 0

    posthog.capture.assert_not_called()
    list_async.assert_awaited_once_with(status="all", limit=100)
    out = capsys.readouterr().out
    assert "DRY RUN, nothing sent" in out
    assert "Stripe: test mode" in out
    assert "Users mapped: 3" in out
    assert "  signed: 1" in out
    assert "  subscribed: 1" in out
    assert "  subscription_ended: 1" in out
    assert "$set sent" not in out


async def test_send_mode_sets_every_user_and_flushes(sources, capsys):
    posthog, _ = sources

    assert await backfill.main(send=True, all_users=False, batch_size=500) == 0

    statuses = {
        call.kwargs["distinct_id"]: call.kwargs["properties"]["$set"][
            "subscription_status"
        ]
        for call in posthog.capture.call_args_list
    }
    assert statuses == {
        "u-1": "subscribed",
        "u-2": "subscription_ended",
        "u-3": "signed",
    }
    assert posthog.flush.called
    assert "$set sent: 3" in capsys.readouterr().out


async def test_send_requires_a_posthog_key(sources, secrets):
    secrets.posthog_api_key = ""
    with pytest.raises(SystemExit):
        await backfill.main(send=True, all_users=False, batch_size=500)


def test_dry_run_is_the_default():
    args = backfill.parse_args([])
    assert args.send is False
    assert args.all_users is False
    assert backfill.parse_args(["--send", "--all-users"]).send is True


@pytest.mark.parametrize("size", ["0", "-5"])
def test_a_non_positive_batch_size_is_rejected(size: str):
    with pytest.raises(SystemExit) as exit_info:
        backfill.parse_args(["--batch-size", size])
    assert exit_info.value.code == 2


def test_a_positive_batch_size_is_accepted():
    assert backfill.parse_args(["--batch-size", "50"]).batch_size == 50
