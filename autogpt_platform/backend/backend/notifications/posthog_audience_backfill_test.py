"""The PostHog backfill gives every account its email type and sign-in method,
and every checkout opener what the live checkout sends, by the same rules,
except that Stripe's first open is set outright."""

from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from backend.notifications import posthog_audience
from backend.notifications import posthog_audience_backfill as backfill
from backend.notifications.audience_enrichment import checkout_fields
from backend.notifications.posthog_audience import CHECKOUT_OPENED_AT, PersonUpdate
from backend.notifications.posthog_audience_backfill import Account, AccountUpdate
from backend.util import posthog_client

CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
OPENED = 1788305400
OPENED_ISO = "2026-09-01T23:30:00+00:00"

ACCOUNTS = [
    Account(
        user_id="billed",
        email="a@acme.io",
        created_at=CREATED,
        timezone="America/Chicago",
        stripe_customer_id="cus_1",
    ),
    Account(
        user_id="opened-only",
        email="b@gmail.com",
        created_at=CREATED,
        timezone="Europe/Vienna",
        stripe_customer_id="cus_2",
    ),
    Account(
        user_id="customer-without-session",
        email="c@uni.edu",
        created_at=CREATED,
        stripe_customer_id="cus_3",
    ),
    Account(
        user_id="never-a-customer",
        email="d@firma.ru",
        created_at=CREATED,
        timezone="Europe/Moscow",
    ),
]
PROVIDERS = {"billed": ["google", "credential"], "opened-only": ["credential"]}
FIRST_OPENS = {"cus_1": OPENED, "cus_2": OPENED + 60}
BILLING = {"cus_1": "US", "cus_3": "FR"}


def _plan():
    return backfill.plan(ACCOUNTS, PROVIDERS, FIRST_OPENS, BILLING)


def test_every_account_is_planned_and_only_openers_get_the_checkout_fields():
    plan = _plan()
    updates = {change.user_id: change.update for change in plan.updates}
    assert len(updates) == 4
    assert plan.openers == 2

    billed = updates["billed"]
    assert billed.to_set["country_code"] == "US"
    assert billed.to_set["country_source"] == "stripe"
    assert billed.to_set[CHECKOUT_OPENED_AT] == OPENED_ISO
    assert billed.to_set_once == {"exclude_de_at": "no"}

    opened_only = updates["opened-only"]
    assert opened_only.to_set_once["country_code"] == "AT"
    assert opened_only.to_set_once["country_source"] == "timezone"
    assert opened_only.to_set["exclude_de_at"] == "yes"

    # A Stripe customer is not an opener without a Checkout Session.
    assert updates["customer-without-session"].to_set == {"email_type": "education"}
    assert updates["customer-without-session"].to_set_once == {}
    # A signal in Russia keeps an account out of MailerLite, not PostHog.
    assert updates["never-a-customer"].to_set == {"email_type": "business"}


def test_the_plan_counts_what_the_accounts_end_up_with():
    plan = _plan()
    assert plan.email_types == {"business": 2, "personal": 1, "education": 1}
    assert plan.signin_methods == {"google": 1, "email": 1, "none": 2}
    assert plan.country_sources == {"stripe": 1, "timezone": 1}
    assert plan.exclude_de_at == 1


def test_an_opener_gets_what_the_live_checkout_sends():
    """The same values by the same rules; only the first open is set
    outright, since Stripe holds every checkout."""
    account = ACCOUNTS[0]
    live = posthog_audience.checkout_update(
        checkout_fields(
            email=account.email,
            created_at=account.created_at,
            opened_at=OPENED,
            signin_providers=PROVIDERS["billed"],
            timezone=account.timezone,
            stripe_country="US",
        ),
        OPENED,
    )
    planned = next(c.update for c in _plan().updates if c.user_id == "billed")
    assert planned.to_set.pop(CHECKOUT_OPENED_AT) == OPENED_ISO
    assert live.to_set_once.pop(CHECKOUT_OPENED_AT) == OPENED_ISO
    assert planned == live


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    """The run's own PostHog client (`posthog_client.send_each`), never the
    shared one. `client.on_error` is the callback the run gave it."""
    client = Mock()
    shared = Mock()
    monkeypatch.setattr(posthog_client, "get_posthog_client", lambda: shared)

    def new_client(**options) -> Mock:
        client.on_error = options["on_error"]
        return client

    monkeypatch.setattr(posthog_client, "new_posthog_client", new_client)
    yield client
    shared.capture.assert_not_called()


@pytest.mark.asyncio
async def test_each_account_is_one_update_on_the_runs_own_client(client: Mock):
    updates = [
        AccountUpdate(user_id="1", update=PersonUpdate(to_set={"email_type": "x"})),
        AccountUpdate(user_id="2", update=PersonUpdate(to_set_once={"country": "y"})),
    ]
    assert await backfill.send(updates) == 0
    sent = [
        (c.kwargs["distinct_id"], c.kwargs["event"], c.kwargs["properties"])
        for c in client.capture.call_args_list
    ]
    assert [(distinct_id, event) for distinct_id, event, _ in sent] == [
        ("1", "$set"),
        ("2", "$set"),
    ]
    assert sent[0][2]["$set"] == {"email_type": "x"}
    assert sent[1][2]["$set_once"] == {"country": "y"}
    client.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_what_posthog_did_not_take_is_returned(client: Mock):
    """Refused when queued, or in an upload that failed for good."""
    client.capture.side_effect = lambda **event: (
        None if event["distinct_id"] == "0" else "event-uuid"
    )
    client.shutdown.side_effect = lambda: client.on_error(
        RuntimeError("503"), [{"distinct_id": "2"}]
    )
    updates = [
        AccountUpdate(user_id=str(i), update=PersonUpdate(to_set={"email_type": "x"}))
        for i in range(3)
    ]
    assert await backfill.send(updates) == 2


@pytest.mark.asyncio
async def test_sending_without_posthog_fails_before_sending_anything(monkeypatch):
    monkeypatch.setattr(posthog_client, "new_posthog_client", lambda **_: None)
    with pytest.raises(RuntimeError, match="PostHog is not configured"):
        await backfill.send(_plan().updates)
