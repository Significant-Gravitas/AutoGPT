"""A checkout backfill that could not write everyone fails as a whole, after
reporting, so a Job or script never reads a partial run as done. An account
that opted out of marketing reaches every backfill's plan as such, and every
report counts it."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from backend.cli import mailerlite_backfill as cli
from backend.notifications import checkout_backfill, mailerlite_backfill
from backend.notifications import mailerlite_field_backfill as field_backfill
from backend.notifications.mailerlite_backfill import Audience, Subscription
from backend.notifications.mailerlite_field_backfill import Person

CREATED = datetime(2026, 7, 14, tzinfo=UTC)
OPTED_OUT = datetime(2026, 10, 2, tzinfo=UTC)


# The root conftest spins a full test server for every test via an autouse
# session fixture; these never touch it.
@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def test_a_complete_run_reports_and_succeeds(capsys):
    cli._finish_checkout(5, 0, 2)
    out = capsys.readouterr().out
    assert "5 ok, 0 failed, 2 already up to date" in out
    assert "Run the dry run again" in out


def test_a_run_with_failures_reports_then_fails():
    with pytest.raises(
        click.ClickException, match="3 checkout openers were not written"
    ):
        cli._finish_checkout(5, 3, 0)


def test_the_command_exits_non_zero_when_anyone_failed(monkeypatch):
    async def run(*, apply: bool, yes: bool) -> None:
        cli._finish_checkout(4, 1, 0)

    monkeypatch.setattr(cli, "_run_checkout", run)
    result = CliRunner().invoke(
        cli.mailerlite_checkout_backfill_command, ["--apply", "--yes"]
    )
    assert result.exit_code == 1
    assert "4 ok, 1 failed" in result.output
    assert "1 checkout openers were not written" in result.output


def _account(user_id: str, opted_out_at: datetime | None) -> SimpleNamespace:
    """A `User` row as `_people` reads it."""
    return SimpleNamespace(
        id=user_id,
        email=f"{user_id}@example.com",
        createdAt=CREATED,
        stripeCustomerId=f"cus_{user_id}",
        timezone=None,
        marketingOptOutAt=opted_out_at,
    )


@pytest.mark.asyncio
async def test_each_account_carries_its_opt_out_into_the_backfills():
    users = MagicMock(
        find_many=AsyncMock(
            return_value=[_account("out", OPTED_OUT), _account("in", None)]
        )
    )
    trials = MagicMock(find_many=AsyncMock(return_value=[]))
    with (
        patch("prisma.models.User.prisma", return_value=users),
        patch("prisma.models.SubscriptionTrial.prisma", return_value=trials),
    ):
        out, kept = await cli._people(
            {"cus_out": [Subscription(status="active")], "cus_in": []}
        )
    assert out.marketing_opt_out_at == OPTED_OUT
    assert kept.marketing_opt_out_at is None
    assert cli._customer(out).marketing_opt_out_at == OPTED_OUT
    assert cli._customer(kept).marketing_opt_out_at is None


def test_every_report_counts_the_opted_out(capsys):
    person = Person(
        user_id="out",
        email="out@example.com",
        created_at=CREATED,
        subscriptions=[Subscription(status="active")],
        stripe_customer_id="cus_out",
        marketing_opt_out_at=OPTED_OUT,
    )
    cli._report(
        [
            mailerlite_backfill.decide(
                cli._customer(person),
                Audience(tour={}, changelog={}, trial={}),
                trial_enabled=True,
            )
        ],
        unmatched=0,
    )
    cli._report_fields(field_backfill.plan([person], {}, create=False), 1)
    cli._report_checkout(
        checkout_backfill.plan(
            [checkout_backfill.Opener(person=person, opened_at=1788305400)], {}, {}
        ),
        sessions_unlinked=0,
        customers_without_session=0,
    )
    out = capsys.readouterr().out
    assert "skip_opted_out: 1" in out
    assert out.count("opted out of marketing (skipped): 1") == 2
    assert "out@example.com" not in out
