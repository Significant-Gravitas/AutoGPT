"""The PostHog backfill's dry run reports counts only and sends nothing, an
--apply run sends the whole plan, and a run PostHog didn't take in full fails
after reporting."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import click
import pytest
import stripe

from backend.cli import mailerlite_backfill
from backend.cli import posthog_audience_backfill as cli
from backend.data import db
from backend.notifications import posthog_audience_backfill as backfill
from backend.notifications.posthog_audience_backfill import Account
from backend.util import settings

CREATED = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
ACCOUNTS = [
    Account(
        user_id="1",
        email="a@acme.io",
        created_at=CREATED,
        timezone="Europe/Vienna",
        stripe_customer_id="cus_1",
    ),
    Account(user_id="2", email="b@gmail.com", created_at=CREATED),
    Account(user_id="3", email="c@acme.io", created_at=CREATED),
]
SESSIONS = [
    {"customer": "cus_1", "created": 1788305400},
    {"customer": "cus_1", "created": 1788305000},
    {"customer": None, "created": 1788300000},
]


def _settings(stripe_api_key: str = "sk_test", posthog_api_key: str = "phc_test"):
    secrets = SimpleNamespace(
        stripe_api_key=stripe_api_key, posthog_api_key=posthog_api_key
    )
    return lambda: SimpleNamespace(secrets=secrets)


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Stripe, the database and PostHog as the run sees them; returns where
    the plan is sent, which reports nobody failed."""
    # Restored afterwards: the run sets the module-wide key.
    monkeypatch.setattr(stripe, "api_key", stripe.api_key)
    monkeypatch.setattr(settings, "Settings", _settings())
    monkeypatch.setattr(
        mailerlite_backfill, "checkout_sessions", AsyncMock(return_value=SESSIONS)
    )
    monkeypatch.setattr(
        mailerlite_backfill,
        "stripe_billing_countries",
        AsyncMock(return_value={"cus_1": "US"}),
    )
    monkeypatch.setattr(
        mailerlite_backfill,
        "signin_providers_by_account",
        AsyncMock(return_value={"1": ["google"]}),
    )
    monkeypatch.setattr(db, "connect", AsyncMock())
    monkeypatch.setattr(db, "disconnect", AsyncMock())
    monkeypatch.setattr(cli, "_accounts", AsyncMock(return_value=ACCOUNTS))
    send = AsyncMock(return_value=0)
    monkeypatch.setattr(backfill, "send", send)
    return send


@pytest.mark.asyncio
async def test_a_dry_run_reports_counts_and_sends_nothing(wired: AsyncMock, capsys):
    await cli._run(apply=False, yes=False)
    out = capsys.readouterr().out
    assert "Accounts: 3" in out
    assert "email_type: business=2, personal=1" in out
    assert "signin_method: none=2, google=1" in out
    assert "Checkout openers (a Stripe Checkout Session): 1" in out
    assert "country_source: stripe=1" in out
    assert "exclude_de_at yes: 1" in out
    # Not what PostHog will hold: a $set_once value may not land.
    assert "These are the values the run sends" in out
    assert "PostHog: 3 persons" in out
    assert "Dry run: nothing was sent" in out
    assert "acme.io" not in out and "gmail.com" not in out
    wired.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_applied_run_sends_the_whole_plan(wired: AsyncMock, capsys):
    await cli._run(apply=True, yes=True)
    (updates,) = wired.await_args.args
    assert [change.user_id for change in updates] == ["1", "2", "3"]
    opener = updates[0].update
    # The first of the customer's Checkout Sessions, not the latest.
    assert opener.to_set["checkout_opened_at"] == "2026-09-01T23:23:20+00:00"
    assert opener.to_set["country_source"] == "stripe"
    assert "PostHog: 3 sent, 0 failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_run_posthog_did_not_take_in_full_fails_after_reporting(
    wired: AsyncMock, capsys
):
    """So a Job or script never reads a partial run as done."""
    wired.return_value = 2
    with pytest.raises(click.ClickException, match="2 accounts were not sent"):
        await cli._run(apply=True, yes=True)
    assert "PostHog: 1 sent, 2 failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_an_applied_run_without_posthog_fails_before_reading_anything(
    wired: AsyncMock, monkeypatch
):
    listed = AsyncMock(return_value=SESSIONS)
    monkeypatch.setattr(mailerlite_backfill, "checkout_sessions", listed)
    monkeypatch.setattr(settings, "Settings", _settings(posthog_api_key=""))
    with pytest.raises(click.ClickException, match="POSTHOG_API_KEY"):
        await cli._run(apply=True, yes=True)
    listed.assert_not_awaited()
    wired.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_dry_run_needs_no_posthog(wired: AsyncMock, monkeypatch, capsys):
    monkeypatch.setattr(settings, "Settings", _settings(posthog_api_key=""))
    await cli._run(apply=False, yes=False)
    assert "Dry run: nothing was sent" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_run_without_stripe_fails_before_reading_anything(
    wired: AsyncMock, monkeypatch
):
    listed = AsyncMock(return_value=SESSIONS)
    monkeypatch.setattr(mailerlite_backfill, "checkout_sessions", listed)
    monkeypatch.setattr(settings, "Settings", _settings(stripe_api_key=""))
    with pytest.raises(click.ClickException, match="STRIPE_API_KEY"):
        await cli._run(apply=False, yes=False)
    listed.assert_not_awaited()
