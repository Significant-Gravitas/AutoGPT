"""Opening Stripe checkout queues the checkout opener for MailerLite, from
each of the three checkout routes and the completed-checkout webhook, and none
of it can cost the checkout or fail the webhook."""

import inspect
import logging
import re
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.api.features import subscription_trial_routes as trial_routes
from backend.api.features.billing.credits import routes as credit_routes
from backend.api.features.billing.subscriptions import routes as subscription_routes
from backend.api.model import RequestTopUp
from backend.data import checkout_audience
from backend.data.notifications import (
    AudienceAction,
    NotificationResult,
    SubscriberField,
)
from backend.data.subscription_trial_checkout import TrialUnavailable
from backend.notifications import consent, subscriber_fields
from backend.notifications.consent_test import _cached_before_consent
from backend.notifications.mailerlite import pseudonym

EMAIL = "sam@example.com"
CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)


# The root conftest spins a full test server for every test via an autouse
# session fixture. These are unit tests over mocks, so shadow it for this
# module, as backend/data/db_test.py does.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


# ── queueing it ──────────────────────────────────────────────────────────────


@pytest.fixture
def queued(monkeypatch):
    queue = AsyncMock(return_value=NotificationResult(success=True))
    monkeypatch.setattr(subscriber_fields, "queue_audience_change", queue)
    monkeypatch.setattr(
        checkout_audience,
        "get_user_by_id",
        AsyncMock(
            return_value=SimpleNamespace(
                email=EMAIL,
                created_at=CREATED,
                timezone="Europe/Vienna",
                marketing_opt_out_at=None,
            )
        ),
    )
    monkeypatch.setattr(
        checkout_audience, "signin_providers", AsyncMock(return_value=["credential"])
    )
    return queue


@pytest.mark.asyncio
async def test_opening_checkout_queues_the_enriched_change(queued):
    checkout_audience.schedule_checkout_opened("user-1", ip_country="US")
    for task in list(checkout_audience._tasks):
        await task
    event = queued.await_args.args[0]
    assert event.action is AudienceAction.CHECKOUT_OPENED
    assert event.fields[SubscriberField.COUNTRY_CODE] == "US"
    assert event.fields[SubscriberField.COUNTRY_SOURCE] == "ip"
    assert event.fields[SubscriberField.SIGNIN_METHOD] == "email"
    # The IP says US, but the browser sits in Vienna.
    assert event.fields[SubscriberField.EXCLUDE_DE_AT] == "yes"


@pytest.mark.asyncio
async def test_a_failure_never_reaches_the_checkout(queued, monkeypatch):
    monkeypatch.setattr(
        checkout_audience, "get_user_by_id", AsyncMock(side_effect=RuntimeError("db"))
    )
    await checkout_audience.queue_checkout_opened("user-1", ip_country="US")
    queued.assert_not_awaited()


def _opt_out(monkeypatch) -> tuple[AsyncMock, AsyncMock]:
    """The account refused marketing at signup."""
    lookup = AsyncMock(
        return_value=SimpleNamespace(
            email=EMAIL,
            created_at=CREATED,
            timezone="Europe/Vienna",
            marketing_opt_out_at=CREATED,
        )
    )
    providers = AsyncMock(return_value=["credential"])
    monkeypatch.setattr(checkout_audience, "get_user_by_id", lookup)
    monkeypatch.setattr(checkout_audience, "signin_providers", providers)
    return lookup, providers


@pytest.mark.asyncio
async def test_an_opted_out_opener_is_never_queued(queued, monkeypatch, caplog):
    _, providers = _opt_out(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger=consent.__name__):
        await checkout_audience.queue_checkout_opened("user-1", ip_country="US")
    queued.assert_not_awaited()
    providers.assert_not_awaited()
    assert pseudonym(EMAIL) in caplog.text
    assert EMAIL not in caplog.text


@pytest.mark.asyncio
async def test_an_opener_cached_before_the_consent_fields_queues_nothing(
    queued, monkeypatch, caplog
):
    """During a rolling deploy the shared cache can hand back a user pickled by
    the previous release, with no opt-out to read. It is skipped, not reported
    as a failed checkout."""
    monkeypatch.setattr(
        checkout_audience,
        "get_user_by_id",
        AsyncMock(return_value=_cached_before_consent()),
    )
    with caplog.at_level(logging.DEBUG, logger=checkout_audience.__name__):
        await checkout_audience.queue_checkout_opened("user-1", ip_country="US")
    queued.assert_not_awaited()
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


@pytest.mark.asyncio
async def test_an_opted_out_customers_completed_checkout_queues_nothing(
    queued, monkeypatch
):
    lookup, _ = _opt_out(monkeypatch)
    with patch(
        "prisma.models.User.prisma",
        return_value=_user_prisma(SimpleNamespace(id="user-1")),
    ):
        await checkout_audience.record_checkout_completed(
            {
                "customer": "cus_1",
                "created": 1788305400,
                "customer_details": {"address": {"country": "DE"}},
            }
        )
    for task in list(checkout_audience._tasks):
        await task
    lookup.assert_awaited_once_with("user-1")
    queued.assert_not_awaited()


def _user_prisma(user):
    return MagicMock(find_first=AsyncMock(return_value=user))


@pytest.mark.asyncio
async def test_a_completed_checkout_sends_its_billing_country(monkeypatch):
    schedule = MagicMock()
    monkeypatch.setattr(checkout_audience, "schedule_checkout_opened", schedule)
    with patch(
        "prisma.models.User.prisma",
        return_value=_user_prisma(SimpleNamespace(id="user-1")),
    ):
        await checkout_audience.record_checkout_completed(
            {
                "customer": "cus_1",
                "created": 1788305400,
                "customer_details": {"address": {"country": "DE"}},
            }
        )
    schedule.assert_called_once_with(
        "user-1", stripe_country="DE", opened_at=1788305400
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session, user",
    [
        ({"customer": "cus_org"}, None),
        ({"customer": None}, SimpleNamespace(id="user-1")),
    ],
    ids=["no-account", "no-customer"],
)
async def test_a_completed_checkout_without_an_account_is_skipped(
    monkeypatch, session, user
):
    schedule = MagicMock()
    monkeypatch.setattr(checkout_audience, "schedule_checkout_opened", schedule)
    with patch("prisma.models.User.prisma", return_value=_user_prisma(user)):
        await checkout_audience.record_checkout_completed(session)
    schedule.assert_not_called()


@pytest.mark.asyncio
async def test_a_completed_checkout_never_fails_the_webhook(monkeypatch):
    with patch("prisma.models.User.prisma", side_effect=RuntimeError("db down")):
        await checkout_audience.record_checkout_completed({"customer": "cus_1"})


# ── the checkout routes and the webhook ──────────────────────────────────────


@pytest.fixture
def schedule():
    return MagicMock()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, queued", [("https://checkout.stripe.com/x", True), ("", False)]
)
async def test_a_top_up_checkout_queues_its_opener(monkeypatch, schedule, url, queued):
    model = SimpleNamespace(top_up_intent=AsyncMock(return_value=url))
    monkeypatch.setattr(
        credit_routes, "get_credit_model", AsyncMock(return_value=model)
    )
    monkeypatch.setattr(credit_routes, "schedule_checkout_opened", schedule)
    await credit_routes.request_top_up(
        request=RequestTopUp(credit_amount=500),
        user_id="user-1",
        ctx=SimpleNamespace(org_id=None),
        country="US",
    )
    if queued:
        schedule.assert_called_once_with("user-1", ip_country="US")
    else:
        schedule.assert_not_called()


def _trial_settings():
    return SimpleNamespace(
        config=SimpleNamespace(
            frontend_base_url="https://platform.example.com", platform_base_url=""
        )
    )


@pytest.mark.asyncio
async def test_a_trial_checkout_queues_its_opener(monkeypatch, schedule):
    monkeypatch.setattr(trial_routes, "Settings", _trial_settings)
    monkeypatch.setattr(
        trial_routes, "create_trial_checkout", AsyncMock(return_value="https://x")
    )
    monkeypatch.setattr(trial_routes, "schedule_checkout_opened", schedule)
    await trial_routes.start_trial_checkout(
        body=trial_routes.TrialCheckoutRequest(offer_token="a" * 64),
        user_id="user-1",
        country="DE",
    )
    schedule.assert_called_once_with("user-1", ip_country="DE")


@pytest.mark.asyncio
async def test_a_trial_checkout_that_never_opened_queues_nothing(monkeypatch, schedule):
    monkeypatch.setattr(trial_routes, "Settings", _trial_settings)
    monkeypatch.setattr(
        trial_routes,
        "create_trial_checkout",
        AsyncMock(side_effect=TrialUnavailable("taken")),
    )
    monkeypatch.setattr(trial_routes, "schedule_checkout_opened", schedule)
    with pytest.raises(HTTPException):
        await trial_routes.start_trial_checkout(
            body=trial_routes.TrialCheckoutRequest(offer_token="a" * 64),
            user_id="user-1",
            country="DE",
        )
    schedule.assert_not_called()


def _source(function) -> str:
    return re.sub(r"\s+", "", inspect.getsource(function))


def test_the_subscription_checkout_and_the_webhook_are_wired():
    """Both need the full stack to drive end to end, so pin the calls: the
    subscription route queues an opener once its session exists, and the
    completed-checkout webhook sends the billing country."""
    route = _source(subscription_routes.update_subscription_tier)
    assert (
        "ifurl:checkout_audience.schedule_checkout_opened(user_id,ip_country=country)"
        in route
    )
    assert route.index("create_subscription_checkout(") < route.index(
        "schedule_checkout_opened"
    )
    webhook = _source(subscription_routes.stripe_webhook)
    assert "awaitcheckout_audience.record_checkout_completed(data_object)" in webhook
